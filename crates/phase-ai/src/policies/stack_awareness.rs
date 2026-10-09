use engine::types::ability::{Effect, QuantityExpr, TargetRef};
use engine::types::actions::GameAction;
use engine::types::game_state::{GameState, StackEntry, StackEntryKind};
use engine::types::identifiers::ObjectId;
use engine::types::keywords::Keyword;
use engine::types::player::PlayerId;
use engine::types::zones::Zone;

use engine::types::card_type::CoreType;

use crate::ability_chain::ChainNode;
use crate::card_value::intrinsic_value;
use crate::eval::{evaluate_creature, evaluate_creature_intrinsic};
use crate::features::DeckFeatures;

use super::activation::turn_only;
use super::context::{collect_ability_effects, collect_resolved_abilities, PolicyContext};
use super::effect_classify::{
    effect_polarity, flicker_target_outcome, is_spell_beneficial, returning_player, EffectPolarity,
    FlickerTargetOutcome,
};
use super::registry::{DecisionKind, PolicyId, PolicyReason, PolicyVerdict, TacticalPolicy};

pub struct StackAwarenessPolicy;

impl StackAwarenessPolicy {
    pub fn score(&self, ctx: &PolicyContext<'_>) -> f64 {
        match &ctx.candidate.action {
            GameAction::ChooseTarget {
                target: Some(TargetRef::Object(id)),
            } => score_target(ctx, *id),
            GameAction::SelectTargets { targets } => targets
                .iter()
                .map(|t| match t {
                    TargetRef::Object(id) => score_target(ctx, *id),
                    _ => 0.0,
                })
                .sum(),
            _ => 0.0,
        }
    }
}

impl TacticalPolicy for StackAwarenessPolicy {
    fn id(&self) -> PolicyId {
        PolicyId::StackAwareness
    }

    fn decision_kinds(&self) -> &'static [DecisionKind] {
        &[DecisionKind::SelectTarget]
    }

    fn activation(
        &self,
        features: &DeckFeatures,
        state: &GameState,
        _player: PlayerId,
    ) -> Option<f32> {
        turn_only(features, state)
    }

    fn verdict(&self, ctx: &PolicyContext<'_>) -> PolicyVerdict {
        PolicyVerdict::Score {
            delta: self.score(ctx),
            reason: PolicyReason::new("stack_awareness_score"),
        }
    }
}

fn score_target(ctx: &PolicyContext<'_>, target_id: ObjectId) -> f64 {
    score_target_redundancy(ctx, target_id)
        + score_counter_target_value(ctx, target_id)
        + score_pump_response(ctx, target_id)
}

fn score_target_redundancy(ctx: &PolicyContext<'_>, target_id: ObjectId) -> f64 {
    if is_spell_beneficial(ctx) {
        return 0.0;
    }

    if !has_pending_removal(ctx.state, target_id) {
        return 0.0;
    }

    if will_target_die_from_stack(ctx.state, target_id) {
        0.0
    } else {
        // Pending removal that might not kill — still penalize but less
        ctx.penalties().redundant_damage_penalty * 0.5
    }
}

/// Impact at which a stack entry is worth the AI's best counter, on the
/// [`assess_spell_impact`] scale. Two consumers share it: the last-counter
/// reservation in [`score_counter_target_value`] (below this, spending the AI's
/// only counter is penalized) and the cast-time impact scaling in
/// `effect_timing::counterspell_score` (at or above this, the counter cast keeps
/// its full stack-pressure bracket).
pub(crate) const COUNTER_IMPACT_THRESHOLD: f64 = 3.0;

/// Impact below which countering is card disadvantage: a counter trades exactly
/// one card, so a target worth less than one card is not worth casting it.
/// Used by `effect_timing::counterspell_score` as the floor of the impact ramp.
///
/// This is a *pricing* boundary, not a card-quality judgement: a 1/1 mana dork
/// prices at 1.05 (0.3 mana value + 0.75 body) and clears it, while Birds of
/// Paradise at 0.6 (0.3 + 0.3 for a 0/1 body) holds. Repricing belongs in
/// [`assess_spell_impact`], not in this constant.
pub(crate) const COUNTER_BREAK_EVEN_IMPACT: f64 = 1.0;

/// CR 701.6a: countering removes a spell from the stack. If `entry` is itself a
/// counter, report what countering *it* is worth to `ai_player`: the impact of
/// the most valuable AI-controlled stack object it targets, or `0.0` when it
/// targets nothing the AI controls (a rival counter aimed at a third player's
/// spell is someone else's fight — countering it buys the AI nothing).
///
/// Returns `None` when `entry` is not a counter at all, so callers can fall back
/// to [`assess_spell_impact`]. Single authority for the "which spell does this
/// foreign counter threaten" walk, shared with `effect_timing`.
pub(crate) fn foreign_counter_target_of_ai(
    state: &GameState,
    entry: &StackEntry,
    ai_player: PlayerId,
) -> Option<f64> {
    let ability = entry.ability()?;
    let effects = collect_ability_effects(ability);
    if !effects.iter().any(|e| matches!(e, Effect::Counter { .. })) {
        return None;
    }

    let mut worth = 0.0_f64;
    for target in &ability.targets {
        let TargetRef::Object(target_id) = target else {
            continue;
        };
        if let Some(threatened) = state.stack.iter().find(|e| e.id == *target_id) {
            if threatened.controller == ai_player {
                worth = worth.max(assess_spell_impact(state, threatened));
            }
        }
    }
    Some(worth)
}

/// When the AI is casting a counter spell, score the target stack entry by its
/// impact. Higher-value spells (by mana value, creature stats, effects) should be
/// preferred counter targets. Returns 0.0 if the pending spell is not a counter.
fn score_counter_target_value(ctx: &PolicyContext<'_>, target_id: ObjectId) -> f64 {
    // Only applies when the AI's pending spell has a Counter effect
    let is_counter = ctx
        .effects()
        .iter()
        .any(|e| matches!(e, Effect::Counter { .. }));
    if !is_counter {
        return 0.0;
    }

    // Find the stack entry being targeted
    let Some(entry) = ctx.state.stack.iter().find(|e| e.id == target_id) else {
        return 0.0;
    };

    // Not a legitimate counter target: countering your own spell is almost always
    // wrong, and so is countering a rival's counter that is aimed at a third
    // player's spell — that resolves someone else's fight at the AI's expense.
    if entry.controller == ctx.ai_player
        || matches!(
            foreign_counter_target_of_ai(ctx.state, entry, ctx.ai_player),
            Some(worth) if worth <= 0.0
        )
    {
        return -10.0;
    }

    let mut score = assess_spell_impact(ctx.state, entry);

    // Last-counter reservation: if this is the AI's only counterspell, penalize
    // spending it on low-impact targets. Save it for something that matters.
    if score < COUNTER_IMPACT_THRESHOLD {
        let counters_in_hand =
            super::strategy_helpers::count_counterspells_in_hand(ctx.state, ctx.ai_player);
        if counters_in_hand == 1 {
            score += ctx.penalties().counter_last_reservation_penalty;
        }
    }

    // Low-MV creature penalty: scale by counter density in hand.
    // With many counters, save them for high-impact threats. With few counters,
    // the current threat IS the thing to counter — don't hold out for something better.
    if let Some(obj) = ctx.state.objects.get(&entry.source_id) {
        let is_cheap_creature = obj.mana_cost.mana_value() <= 2
            && obj
                .card_types
                .core_types
                .contains(&engine::types::card_type::CoreType::Creature);
        if is_cheap_creature {
            let intent = crate::eval::strategic_intent(ctx.state, ctx.ai_player);
            if !matches!(intent, crate::eval::StrategicIntent::Stabilize) {
                let counters_in_hand =
                    super::strategy_helpers::count_counterspells_in_hand(ctx.state, ctx.ai_player);
                // 1 counter = no penalty, 2 = -0.3, 3+ = -0.6
                let penalty = -0.3 * (counters_in_hand as f64 - 1.0).clamp(0.0, 2.0);
                score += penalty;
            }
        }
    }

    score
}

/// When the AI's pending spell is harmful, boost targeting a creature that an
/// opponent is currently pumping on the stack — removing it wastes both the
/// creature and the pump spell (2-for-1).
fn score_pump_response(ctx: &PolicyContext<'_>, target_id: ObjectId) -> f64 {
    if is_spell_beneficial(ctx) {
        return 0.0;
    }

    // Skip if target is already dying — redundancy penalty handles that case
    if will_target_die_from_stack(ctx.state, target_id) {
        return 0.0;
    }

    let has_opponent_pump = ctx.state.stack.iter().any(|entry| {
        entry.controller != ctx.ai_player && {
            let Some(ability) = entry.ability() else {
                return false;
            };
            let targets_this = ability
                .targets
                .iter()
                .any(|t| matches!(t, TargetRef::Object(id) if *id == target_id));
            targets_this
                && collect_ability_effects(ability)
                    .iter()
                    .any(|e| matches!(e, Effect::Pump { .. } | Effect::DoublePT { .. }))
        }
    });

    if has_opponent_pump {
        ctx.penalties().pump_response_bonus
    } else {
        0.0
    }
}

/// Estimate the game impact of a stack entry based on its effects.
/// Used for counter-target valuation and protect-my-spell incentives.
pub(crate) fn assess_spell_impact(state: &GameState, entry: &StackEntry) -> f64 {
    match &entry.kind {
        StackEntryKind::Spell { .. } => {
            let mv = state
                .objects
                .get(&entry.source_id)
                .map(|o| o.mana_cost.mana_value())
                .unwrap_or(0) as f64;

            let mut score = mv * 0.3;

            let abilities = entry
                .ability()
                .map(collect_resolved_abilities)
                .unwrap_or_default();
            for ability in abilities {
                score += match &ability.effect {
                    Effect::ExtraTurn { count, .. } => {
                        engine::game::quantity::resolve_quantity_with_targets(state, count, ability)
                            .max(0) as f64
                            * 5.0
                    }
                    Effect::DestroyAll { .. }
                    | Effect::DamageAll { .. }
                    | Effect::ChangeZoneAll { .. } => 4.0,
                    Effect::GainControl { .. } | Effect::GainControlAll { .. } => 2.5,
                    Effect::Destroy { .. } | Effect::Fight { .. } => 1.5,
                    Effect::Counter { .. } => 1.5,
                    Effect::Draw {
                        count: QuantityExpr::Fixed { value },
                        ..
                    } => *value as f64 * 1.5,
                    Effect::DealDamage { .. } => 1.0,
                    Effect::SearchLibrary { .. } => 1.0,
                    Effect::Token { .. } => 0.5,
                    _ => 0.0,
                };
            }

            // Creature spells: factor in the creature's board value
            let creature_value = evaluate_creature(state, entry.source_id);
            if creature_value > 0.0 {
                score += creature_value * 0.3;
            }

            score.min(8.0)
        }
        // Activated/triggered abilities: moderate value — they're free to re-trigger.
        // KeywordAction (Crew/Equip/Saddle/Station) is similarly low-value to counter:
        // the cost was paid at announcement and the activation can be repeated.
        StackEntryKind::ActivatedAbility { .. }
        | StackEntryKind::TriggeredAbility { .. }
        | StackEntryKind::KeywordAction { .. } => 0.5,
        // Combat damage on the stack is neither a spell nor an ability, so no
        // counter can target it (CR 112.1 + CR 113.3b) and no protect-my-spell
        // incentive applies. This function values COUNTER TARGETS, so an entry
        // that can never be one is worth nothing to it.
        StackEntryKind::CombatDamage { .. } => 0.0,
    }
}

/// Check if any stack entry targets this object with a harmful effect.
pub(crate) fn has_pending_removal(state: &GameState, target_id: ObjectId) -> bool {
    state.stack.iter().any(|entry| {
        let Some(ability) = entry.ability() else {
            return false;
        };
        let targets_this = ability
            .targets
            .iter()
            .any(|t| matches!(t, TargetRef::Object(id) if *id == target_id));
        if !targets_this {
            return false;
        }
        // Check if any effect in the chain is harmful
        collect_ability_effects(ability)
            .iter()
            .any(|e| matches!(effect_polarity(e), EffectPolarity::Harmful))
    })
}

/// Estimate whether pending stack effects will remove this object (creature or spell).
pub(crate) fn will_target_die_from_stack(state: &GameState, target_id: ObjectId) -> bool {
    entries_remove_target(state, target_id, |_| true)
}

/// CR 608.2b + CR 400.7: a spell or ability that targets this object and
/// would remove it, whoever controls it; flickering the object first leaves
/// that target illegal, so the removal does nothing.
///
/// CR 400.7 + CR 102.3: an exile-and-return that hands the object straight
/// back, or that the AI or a teammate controls and the flicker authority
/// reads as a re-entry, is not removal. CR 115.10a: untargeted mass removal
/// and player-targeted edicts never target the object, so they are not
/// answered.
pub(crate) fn pending_removal_will_remove(
    state: &GameState,
    ai_player: PlayerId,
    target_id: ObjectId,
) -> bool {
    entries_remove_target(state, target_id, |entry| {
        !entry_flickers_target_back(state, entry, target_id, ai_player)
    })
}

/// True when some node of `entry`'s resolved chain targets `target_id` with a
/// flicker pair that brings it back rather than removing it:
/// (a) an immediate return that puts it back under `ai_player`'s control,
/// whoever controls the entry (an opposing Eldrazi Displacer on the AI's own
/// creature); or
/// (b) a pair the flicker authority reads, from the entry controller's side,
/// as a re-entry of the object (the AI's own Guardian of Ghirapur trigger, a
/// teammate's Flickerwisp trigger) at any return timing. For an opponent's
/// entry the authority reads `OpposingObject` first, so (b) never excuses an
/// opposing flicker.
fn entry_flickers_target_back(
    state: &GameState,
    entry: &StackEntry,
    target_id: ObjectId,
    ai_player: PlayerId,
) -> bool {
    let (Some(ability), Some(object)) = (entry.ability(), state.objects.get(&target_id)) else {
        return false;
    };
    collect_resolved_abilities(ability).into_iter().any(|node| {
        node.targets
            .iter()
            .any(|target| matches!(target, TargetRef::Object(id) if *id == target_id))
            && ChainNode::Resolved(node)
                .flicker_pair()
                .is_some_and(|pair| {
                    (pair.returns.is_immediate()
                        && returning_player(&pair, object, entry.controller) == Some(ai_player))
                        || flicker_target_outcome(state, &pair, target_id, entry.controller)
                            == Some(FlickerTargetOutcome::Reenters)
                })
    })
}

/// What losing permanent `id` costs its controller: a creature's intrinsic
/// value (without the tapped discount — a tapped creature is lost just the
/// same), otherwise the card's intrinsic value. Never negative.
pub(crate) fn threatened_permanent_value(state: &GameState, id: ObjectId) -> f64 {
    let Some(object) = state.objects.get(&id) else {
        return 0.0;
    };
    let value = if object.card_types.core_types.contains(&CoreType::Creature) {
        evaluate_creature_intrinsic(state, id)
    } else {
        intrinsic_value(state, id)
    };
    value.max(0.0)
}

/// Whether the stack entries `keep_entry` admits will remove `target_id`
/// (creature or spell): a destroy, counter, bounce or zone change away that
/// targets it, or targeted fixed damage that adds up to lethal.
fn entries_remove_target(
    state: &GameState,
    target_id: ObjectId,
    keep_entry: impl Fn(&StackEntry) -> bool,
) -> bool {
    let Some(object) = state.objects.get(&target_id) else {
        return false;
    };

    let mut pending_damage: i32 = 0;

    for entry in state.stack.iter() {
        if !keep_entry(entry) {
            continue;
        }
        let Some(ability) = entry.ability() else {
            continue;
        };
        let targets_this = ability
            .targets
            .iter()
            .any(|t| matches!(t, TargetRef::Object(id) if *id == target_id));
        if !targets_this {
            continue;
        }

        for effect in collect_ability_effects(ability) {
            match effect {
                // Destroy is lethal unless target is indestructible
                Effect::Destroy { .. } if !object.has_keyword(&Keyword::Indestructible) => {
                    return true;
                }
                // Counter removes the spell from the stack
                Effect::Counter { .. } => return true,
                // Bounce removes from battlefield
                Effect::Bounce { .. } => return true,
                // ChangeZone to non-battlefield removes from battlefield
                Effect::ChangeZone {
                    destination: Zone::Exile | Zone::Graveyard | Zone::Hand | Zone::Library,
                    ..
                } => {
                    return true;
                }
                // Accumulate pending damage
                Effect::DealDamage {
                    amount: QuantityExpr::Fixed { value },
                    ..
                } => {
                    pending_damage += value;
                }
                _ => {}
            }
        }
    }

    // Check if accumulated pending damage is lethal
    if let Some(toughness) = object.toughness {
        let remaining = toughness - object.damage_marked as i32;
        pending_damage >= remaining
    } else {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::AiConfig;
    use engine::ai_support::{ActionMetadata, AiDecisionContext, CandidateAction, TacticalClass};
    use engine::game::zones::create_object;
    use engine::types::ability::{
        BounceSelection, EffectKind, QuantityRef, ResolvedAbility, TargetFilter,
    };
    use engine::types::card_type::CoreType;
    use engine::types::game_state::{
        GameState, PendingCast, StackEntry, StackEntryKind, TargetEffectDetail,
        TargetSelectionSlot, WaitingFor,
    };
    use engine::types::identifiers::{CardId, ObjectId};
    use engine::types::mana::ManaCost;
    use engine::types::player::PlayerId;
    use engine::types::zones::Zone;

    fn make_state() -> GameState {
        let mut state = GameState::new_two_player(42);
        state.turn_number = 2;
        state
    }

    fn add_creature(
        state: &mut GameState,
        owner: PlayerId,
        power: i32,
        toughness: i32,
    ) -> ObjectId {
        let id = create_object(
            state,
            CardId(state.next_object_id),
            owner,
            "Creature".to_string(),
            Zone::Battlefield,
        );
        let obj = state.objects.get_mut(&id).unwrap();
        obj.card_types.core_types.push(CoreType::Creature);
        obj.power = Some(power);
        obj.toughness = Some(toughness);
        id
    }

    fn push_stack_entry(state: &mut GameState, effect: Effect, targets: Vec<TargetRef>) {
        let ability = ResolvedAbility::new(effect, targets, ObjectId(999), PlayerId(1));
        state.stack.push_back(StackEntry {
            id: ObjectId(state.next_object_id),
            source_id: ObjectId(999),
            controller: PlayerId(1),
            kind: StackEntryKind::Spell {
                ability: Some(Box::new(ability)),
                card_id: CardId(999),
                casting_variant: Default::default(),
                actual_mana_spent: 0,
            },
        });
        state.next_object_id += 1;
    }

    fn counter_effect() -> Effect {
        Effect::Counter {
            target: TargetFilter::Any,
            source_rider: None,
            countered_spell_zone: None,
        }
    }

    fn extra_turn(count: QuantityExpr) -> Effect {
        Effect::ExtraTurn {
            target: TargetFilter::Controller,
            count,
        }
    }

    #[test]
    fn spell_impact_values_each_extra_turn_and_uses_the_owning_node_context() {
        let mut state = make_state();
        let unrelated_id = push_spell(&mut state, PlayerId(1), 0, Effect::NoOp, vec![]);
        let unrelated = state
            .stack
            .iter()
            .find(|entry| entry.id == unrelated_id)
            .unwrap();
        assert_eq!(assess_spell_impact(&state, unrelated), 0.0);

        let one_id = push_spell(
            &mut state,
            PlayerId(1),
            0,
            extra_turn(QuantityExpr::Fixed { value: 1 }),
            vec![],
        );
        let one = state.stack.iter().find(|entry| entry.id == one_id).unwrap();
        assert_eq!(assess_spell_impact(&state, one), 5.0);

        let two_id = push_spell(
            &mut state,
            PlayerId(1),
            0,
            extra_turn(QuantityExpr::Fixed { value: 2 }),
            vec![],
        );
        let two = state.stack.iter().find(|entry| entry.id == two_id).unwrap();
        assert_eq!(assess_spell_impact(&state, two), 8.0);

        let source_card_id = CardId(state.next_object_id);
        let source_id = create_object(
            &mut state,
            source_card_id,
            PlayerId(1),
            "Context Spell".into(),
            Zone::Stack,
        );
        let mut root = ResolvedAbility::new(Effect::NoOp, vec![], source_id, PlayerId(1));
        root.chosen_x = Some(1);
        let mut child = ResolvedAbility::new(
            extra_turn(QuantityExpr::Ref {
                qty: QuantityRef::Variable { name: "X".into() },
            }),
            vec![],
            source_id,
            PlayerId(1),
        );
        child.chosen_x = Some(2);
        root.sub_ability = Some(Box::new(child));
        let mut root_one = root.clone();
        root_one
            .sub_ability
            .as_mut()
            .expect("dynamic child")
            .chosen_x = Some(1);
        let entry = StackEntry {
            id: ObjectId(state.next_object_id),
            source_id,
            controller: PlayerId(1),
            kind: StackEntryKind::Spell {
                ability: Some(Box::new(root)),
                card_id: CardId(state.next_object_id),
                casting_variant: Default::default(),
                actual_mana_spent: 0,
            },
        };
        assert_eq!(assess_spell_impact(&state, &entry), 8.0);
        let one_entry = StackEntry {
            id: ObjectId(state.next_object_id + 1),
            source_id,
            controller: PlayerId(1),
            kind: StackEntryKind::Spell {
                ability: Some(Box::new(root_one)),
                card_id: CardId(state.next_object_id + 1),
                casting_variant: Default::default(),
                actual_mana_spent: 0,
            },
        };
        assert_eq!(assess_spell_impact(&state, &one_entry), 5.0);
    }

    /// Sibling of [`push_stack_entry`] for multiplayer counter fixtures: the entry
    /// carries an explicit `controller` and is backed by a real object, so
    /// [`assess_spell_impact`] can read its mana value. Returns the entry id.
    fn push_spell(
        state: &mut GameState,
        controller: PlayerId,
        mana_value: u32,
        effect: Effect,
        targets: Vec<TargetRef>,
    ) -> ObjectId {
        let source_id = create_object(
            state,
            CardId(state.next_object_id),
            controller,
            "Spell".to_string(),
            Zone::Stack,
        );
        state.objects.get_mut(&source_id).unwrap().mana_cost = ManaCost::generic(mana_value);
        let ability = ResolvedAbility::new(effect, targets, source_id, controller);
        let id = ObjectId(state.next_object_id);
        state.next_object_id += 1;
        state.stack.push_back(StackEntry {
            id,
            source_id,
            controller,
            kind: StackEntryKind::Spell {
                ability: Some(Box::new(ability)),
                card_id: CardId(id.0),
                casting_variant: Default::default(),
                actual_mana_spent: 0,
            },
        });
        id
    }

    fn make_target_ctx(
        _state: &GameState,
        target_id: ObjectId,
        source_effect: Effect,
    ) -> (AiDecisionContext, CandidateAction) {
        let ability = ResolvedAbility::new(source_effect, Vec::new(), ObjectId(888), PlayerId(1));
        let pending_cast = PendingCast::new(ObjectId(888), CardId(888), ability, ManaCost::zero());
        let decision = AiDecisionContext {
            waiting_for: WaitingFor::TargetSelection {
                player: PlayerId(1),
                pending_cast: Box::new(pending_cast),
                target_slots: vec![TargetSelectionSlot {
                    legal_targets: vec![TargetRef::Object(target_id)],
                    optional: false,
                    chooser: None,
                    effect_kind: EffectKind::NoOp,
                    effect_detail: TargetEffectDetail::None,
                }],
                mode_labels: Vec::new(),
                selection: Default::default(),
            },
            candidates: Vec::new(),
        };
        let candidate = CandidateAction {
            action: GameAction::ChooseTarget {
                target: Some(TargetRef::Object(target_id)),
            },
            metadata: ActionMetadata::for_actor(Some(PlayerId(1)), TacticalClass::Target),
        };
        (decision, candidate)
    }

    fn score_policy(
        state: &GameState,
        decision: &AiDecisionContext,
        candidate: &CandidateAction,
    ) -> f64 {
        let config = AiConfig::default();
        let ctx = PolicyContext {
            state,
            decision,
            candidate,
            ai_player: PlayerId(1),
            config: &config,
            context: &crate::context::AiContext::empty(&config.weights),
            cast_facts: None,
            search_depth: crate::policies::context::SearchDepth::Root,
        };
        StackAwarenessPolicy.score(&ctx)
    }

    // --- Helper tests ---

    #[test]
    fn has_pending_removal_finds_destroy() {
        let mut state = make_state();
        let creature = add_creature(&mut state, PlayerId(0), 3, 3);
        push_stack_entry(
            &mut state,
            Effect::Destroy {
                target: TargetFilter::Any,
                cant_regenerate: false,
            },
            vec![TargetRef::Object(creature)],
        );
        assert!(has_pending_removal(&state, creature));
    }

    #[test]
    fn has_pending_removal_ignores_different_target() {
        let mut state = make_state();
        let creature_a = add_creature(&mut state, PlayerId(0), 3, 3);
        let creature_b = add_creature(&mut state, PlayerId(0), 2, 2);
        push_stack_entry(
            &mut state,
            Effect::Destroy {
                target: TargetFilter::Any,
                cant_regenerate: false,
            },
            vec![TargetRef::Object(creature_a)],
        );
        assert!(!has_pending_removal(&state, creature_b));
    }

    #[test]
    fn will_target_die_destroy() {
        let mut state = make_state();
        let creature = add_creature(&mut state, PlayerId(0), 3, 3);
        push_stack_entry(
            &mut state,
            Effect::Destroy {
                target: TargetFilter::Any,
                cant_regenerate: false,
            },
            vec![TargetRef::Object(creature)],
        );
        assert!(will_target_die_from_stack(&state, creature));
    }

    #[test]
    fn will_target_die_indestructible_survives_destroy() {
        let mut state = make_state();
        let creature = add_creature(&mut state, PlayerId(0), 3, 3);
        state
            .objects
            .get_mut(&creature)
            .unwrap()
            .keywords
            .push(Keyword::Indestructible);
        push_stack_entry(
            &mut state,
            Effect::Destroy {
                target: TargetFilter::Any,
                cant_regenerate: false,
            },
            vec![TargetRef::Object(creature)],
        );
        assert!(!will_target_die_from_stack(&state, creature));
    }

    #[test]
    fn will_target_die_lethal_damage() {
        let mut state = make_state();
        let creature = add_creature(&mut state, PlayerId(0), 2, 3);
        push_stack_entry(
            &mut state,
            Effect::DealDamage {
                amount: QuantityExpr::Fixed { value: 3 },
                target: TargetFilter::Any,
                damage_source: None,
                excess: None,
            },
            vec![TargetRef::Object(creature)],
        );
        assert!(will_target_die_from_stack(&state, creature));
    }

    #[test]
    fn will_target_die_insufficient_damage() {
        let mut state = make_state();
        let creature = add_creature(&mut state, PlayerId(0), 2, 4);
        push_stack_entry(
            &mut state,
            Effect::DealDamage {
                amount: QuantityExpr::Fixed { value: 2 },
                target: TargetFilter::Any,
                damage_source: None,
                excess: None,
            },
            vec![TargetRef::Object(creature)],
        );
        assert!(!will_target_die_from_stack(&state, creature));
    }

    #[test]
    fn will_target_die_bounce() {
        let mut state = make_state();
        let creature = add_creature(&mut state, PlayerId(0), 3, 3);
        push_stack_entry(
            &mut state,
            Effect::Bounce {
                target: TargetFilter::Any,
                destination: None,
                selection: BounceSelection::Targeted,
            },
            vec![TargetRef::Object(creature)],
        );
        assert!(will_target_die_from_stack(&state, creature));
    }

    // --- Policy-level tests ---

    #[test]
    fn no_penalty_when_different_targets() {
        let mut state = make_state();
        let creature_a = add_creature(&mut state, PlayerId(0), 3, 3);
        let creature_b = add_creature(&mut state, PlayerId(0), 2, 2);
        push_stack_entry(
            &mut state,
            Effect::Destroy {
                target: TargetFilter::Any,
                cant_regenerate: false,
            },
            vec![TargetRef::Object(creature_a)],
        );

        let (decision, candidate) = make_target_ctx(
            &state,
            creature_b,
            Effect::Destroy {
                target: TargetFilter::Any,
                cant_regenerate: false,
            },
        );
        let score = score_policy(&state, &decision, &candidate);
        assert!(
            score.abs() < 0.01,
            "No penalty when targeting different creature, got {score}"
        );
    }

    #[test]
    fn empty_stack_no_penalty() {
        let mut state = make_state();
        let creature = add_creature(&mut state, PlayerId(0), 3, 3);

        let (decision, candidate) = make_target_ctx(
            &state,
            creature,
            Effect::Destroy {
                target: TargetFilter::Any,
                cant_regenerate: false,
            },
        );
        let score = score_policy(&state, &decision, &candidate);
        assert!(
            score.abs() < 0.01,
            "No penalty with empty stack, got {score}"
        );
    }

    #[test]
    fn indestructible_not_penalized_second_removal() {
        let mut state = make_state();
        let creature = add_creature(&mut state, PlayerId(0), 3, 3);
        state
            .objects
            .get_mut(&creature)
            .unwrap()
            .keywords
            .push(Keyword::Indestructible);
        // First Destroy won't kill it (indestructible)
        push_stack_entry(
            &mut state,
            Effect::Destroy {
                target: TargetFilter::Any,
                cant_regenerate: false,
            },
            vec![TargetRef::Object(creature)],
        );

        // Second removal should still get partial penalty (there IS pending removal,
        // just not lethal)
        let (decision, candidate) = make_target_ctx(
            &state,
            creature,
            Effect::DealDamage {
                amount: QuantityExpr::Fixed { value: 5 },
                target: TargetFilter::Any,
                damage_source: None,
                excess: None,
            },
        );
        let score = score_policy(&state, &decision, &candidate);
        // Should get partial penalty (redundant_damage * 0.5), not full redundant_removal
        assert!(
            score < 0.0 && score > -5.0,
            "Should get partial penalty for indestructible, got {score}"
        );
    }

    /// Three players: B (P0) counters `victim_controller`'s spell. When the victim
    /// is C (P2), countering B's counter only resolves C's spell — the AI (P1)
    /// would be fighting someone else's fight for a card. Returns B's entry id.
    fn three_player_counter_war(victim_controller: PlayerId) -> (GameState, ObjectId) {
        let mut state = GameState::new(engine::types::format::FormatConfig::free_for_all(), 3, 42);
        state.turn_number = 2;
        let victim = push_spell(&mut state, victim_controller, 4, Effect::NoOp, Vec::new());
        let rival_counter = push_spell(
            &mut state,
            PlayerId(0),
            2,
            counter_effect(),
            vec![TargetRef::Object(victim)],
        );
        (state, rival_counter)
    }

    #[test]
    fn counter_target_rival_counter_on_third_party_spell_is_penalised() {
        let (state, rival_counter) = three_player_counter_war(PlayerId(2));
        let (decision, candidate) = make_target_ctx(&state, rival_counter, counter_effect());
        let score = score_policy(&state, &decision, &candidate);
        assert!(
            score <= -10.0,
            "A rival counter aimed at a third player's spell must not be a counter \
             target, got {score}"
        );
    }

    #[test]
    fn counter_target_rival_counter_on_own_spell_is_valued() {
        // PlayerId(1) is the AI seat that `score_policy` scores from.
        let (state, rival_counter) = three_player_counter_war(PlayerId(1));
        let (decision, candidate) = make_target_ctx(&state, rival_counter, counter_effect());
        let score = score_policy(&state, &decision, &candidate);
        assert!(
            score > 0.0,
            "A rival counter aimed at the AI's own spell is a legitimate counter \
             target, got {score}"
        );
    }
}

/// Phase-1 flicker rows U-S1 and U-S2: the pending-removal predicate reads
/// the charter's r9 rescue reading.
#[cfg(test)]
mod flicker_rows {
    use engine::game::ability_utils::build_resolved_from_def_with_targets;
    use engine::game::scenario::{GameScenario, P0, P1};
    use engine::types::ability::AbilityDefinition;
    use engine::types::format::FormatConfig;
    use engine::types::game_state::{StackEntry, StackEntryKind};
    use engine::types::identifiers::{CardId, ObjectId};

    use super::*;
    use crate::policies::context::flicker_fixtures as fx;

    #[derive(Clone, Copy)]
    enum EntryKind {
        Spell,
        Activated,
        Triggered,
    }

    /// Put `root` on the stack as `controller`'s `kind` entry from `source`,
    /// targeting `targets` (the parsed card's own ability, as the engine
    /// resolves it).
    fn push_entry(
        state: &mut GameState,
        controller: PlayerId,
        source: ObjectId,
        kind: EntryKind,
        root: &AbilityDefinition,
        targets: Vec<TargetRef>,
    ) {
        let ability = Box::new(build_resolved_from_def_with_targets(
            root, source, controller, targets,
        ));
        let id = ObjectId(state.next_object_id);
        state.next_object_id += 1;
        let kind = match kind {
            EntryKind::Spell => StackEntryKind::Spell {
                card_id: CardId(id.0),
                ability: Some(ability),
                casting_variant: Default::default(),
                actual_mana_spent: 0,
            },
            EntryKind::Activated => StackEntryKind::ActivatedAbility {
                source_id: source,
                ability,
            },
            EntryKind::Triggered => StackEntryKind::TriggeredAbility {
                source_id: source,
                ability,
                condition: None,
                trigger_event: None,
                description: None,
                source_name: String::new(),
                subject_match_count: None,
                die_result: None,
                provenance: None,
            },
        };
        state.stack.push_back(StackEntry {
            id,
            source_id: source,
            controller,
            kind,
        });
    }

    fn root(name: &str, types: &[&str], keywords: &[&str], oracle: &str) -> AbilityDefinition {
        fx::roots(name, types, keywords, oracle).remove(0)
    }

    fn murder() -> AbilityDefinition {
        root("Murder", &["Instant"], &[], fx::MURDER)
    }

    fn flickerwisp() -> AbilityDefinition {
        root("Flickerwisp", &["Creature"], fx::FLYING, fx::FLICKERWISP)
    }

    /// One U-S1 reading: a two-player (or 2HG) board where the AI controls
    /// the object `setup` returns, and one entry on the stack.
    fn reading(
        two_headed: bool,
        setup: impl FnOnce(&mut GameScenario) -> (ObjectId, ObjectId),
        controller: PlayerId,
        kind: EntryKind,
        ability: AbilityDefinition,
        targets: Option<Vec<TargetRef>>,
    ) -> bool {
        let mut scenario = if two_headed {
            GameScenario::new_with_format(FormatConfig::two_headed_giant(), 4, 42)
        } else {
            fx::scenario()
        };
        let (source, object) = setup(&mut scenario);
        let mut runner = scenario.build();
        // `None`: the entry targets the object under test.
        let targets = targets.unwrap_or_else(|| vec![TargetRef::Object(object)]);
        push_entry(
            runner.state_mut(),
            controller,
            source,
            kind,
            &ability,
            targets,
        );
        let removes = pending_removal_will_remove(runner.state(), P0, object);
        removes
    }

    fn v_and_source(
        scenario: &mut GameScenario,
        source_controller: PlayerId,
    ) -> (ObjectId, ObjectId) {
        let v = fx::giant(scenario, P0);
        let source = scenario
            .add_creature(source_controller, "Source", 1, 1)
            .id();
        (source, v)
    }

    // U-S1: a removal counts whoever controls it; an exile-and-return that
    // brings the object back does not.
    #[test]
    fn pending_removal_reads_the_rescue_reading() {
        let rows: Vec<(&str, bool, bool)> = vec![
            (
                "opponent Murder -> V",
                reading(
                    false,
                    |s| v_and_source(s, P1),
                    P1,
                    EntryKind::Spell,
                    murder(),
                    None,
                ),
                true,
            ),
            (
                "AI's own Murder -> V",
                reading(
                    false,
                    |s| v_and_source(s, P0),
                    P0,
                    EntryKind::Spell,
                    murder(),
                    None,
                ),
                true,
            ),
            (
                "2HG teammate Murder -> V",
                reading(
                    true,
                    |s| v_and_source(s, P1),
                    P1,
                    EntryKind::Spell,
                    murder(),
                    None,
                ),
                true,
            ),
            (
                "opponent Flickerwisp trigger -> V",
                reading(
                    false,
                    |s| v_and_source(s, P1),
                    P1,
                    EntryKind::Triggered,
                    flickerwisp(),
                    None,
                ),
                true,
            ),
            (
                "opponent Eldrazi Displacer -> AI-owned V",
                reading(
                    false,
                    |s| v_and_source(s, P1),
                    P1,
                    EntryKind::Activated,
                    root(
                        "Eldrazi Displacer",
                        &["Creature"],
                        fx::ELDRAZI_DISPLACER_KEYWORDS,
                        fx::ELDRAZI_DISPLACER,
                    ),
                    None,
                ),
                false,
            ),
            (
                "AI's own Guardian trigger -> V",
                reading(
                    false,
                    |s| v_and_source(s, P0),
                    P0,
                    EntryKind::Triggered,
                    root(
                        "Guardian of Ghirapur",
                        &["Creature"],
                        fx::FLYING,
                        fx::GUARDIAN_OF_GHIRAPUR,
                    ),
                    None,
                ),
                false,
            ),
            (
                "2HG teammate Flickerwisp trigger -> AI's V",
                reading(
                    true,
                    |s| v_and_source(s, P1),
                    P1,
                    EntryKind::Triggered,
                    flickerwisp(),
                    None,
                ),
                false,
            ),
            (
                "2HG teammate Flickerwisp trigger -> stolen creature",
                reading(
                    true,
                    |s| {
                        let stolen = s
                            .add_creature(PlayerId(2), "Stolen Bear", 2, 2)
                            .controlled_by(P0)
                            .id();
                        let source = s.add_creature(P1, "Source", 1, 1).id();
                        (source, stolen)
                    },
                    P1,
                    EntryKind::Triggered,
                    flickerwisp(),
                    None,
                ),
                true,
            ),
            (
                "opponent Murder -> indestructible V",
                reading(
                    false,
                    |s| {
                        let v = s.add_creature(P0, "Hill Giant", 3, 3).indestructible().id();
                        let source = s.add_creature(P1, "Source", 1, 1).id();
                        (source, v)
                    },
                    P1,
                    EntryKind::Spell,
                    murder(),
                    None,
                ),
                false,
            ),
            (
                "opponent Shock -> 3/3",
                reading(
                    false,
                    |s| v_and_source(s, P1),
                    P1,
                    EntryKind::Spell,
                    root("Shock", &["Instant"], &[], fx::SHOCK),
                    None,
                ),
                false,
            ),
            (
                "opponent Day of Judgment",
                reading(
                    false,
                    |s| v_and_source(s, P1),
                    P1,
                    EntryKind::Spell,
                    root("Day of Judgment", &["Sorcery"], &[], fx::DAY_OF_JUDGMENT),
                    Some(Vec::new()),
                ),
                false,
            ),
            (
                "opponent Diabolic Edict -> AI",
                reading(
                    false,
                    |s| v_and_source(s, P1),
                    P1,
                    EntryKind::Spell,
                    root("Diabolic Edict", &["Instant"], &[], fx::DIABOLIC_EDICT),
                    Some(vec![TargetRef::Player(P0)]),
                ),
                false,
            ),
        ];
        for (label, removes, expected) in rows {
            assert_eq!(removes, expected, "{label}");
        }
    }

    // U-S2: the shared body behaves as before for every entry.
    #[test]
    fn will_target_die_from_stack_unchanged_by_entry_predicate() {
        let mut scenario = fx::scenario();
        let v = fx::giant(&mut scenario, P0);
        let source = scenario.add_creature(P0, "Source", 1, 1).id();
        let mut runner = scenario.build();
        push_entry(
            runner.state_mut(),
            P0,
            source,
            EntryKind::Spell,
            &murder(),
            vec![TargetRef::Object(v)],
        );
        let dies = will_target_die_from_stack(runner.state(), v);
        assert!(dies);
    }
}
