//! Shared helpers for walking an `AbilityDefinition`'s effect tree.
//!
//! `AbilityDefinition` composes a primary `effect` with an optional
//! `sub_ability` that itself has an `effect` plus another optional
//! `sub_ability`, forming a single-linked list. Feature detectors and
//! policies both need to classify the *set* of effects produced by an
//! ability (e.g., "does this ability both search the library and put a
//! land onto the battlefield?"), so they collect the chain into a flat
//! slice and iterate with `matches!`.
//!
//! Two branches of that definition are *conditional* rather than part of the
//! unconditional chain: `else_ability` (the CR 608.2c "Otherwise, ..." leg) and
//! `mode_abilities` (the CR 700.2 modal alternatives). Whether they belong in
//! the walk depends on the question being asked, which is what [`AbilityScope`]
//! names — see its docs. [`collect_scoped_effects`] is the single authority for
//! both walks; `collect_chain_effects` is the unconditional shorthand.
//!
//! Keep this module small — it is a single building block shared across
//! `features/*` and `policies/*`.

use engine::types::ability::{
    AbilityDefinition, ControllerRef, DelayedTriggerCondition, Effect, QuantityExpr,
    ResolvedAbility, TargetFilter,
};
use engine::types::counter::CounterType;
use engine::types::zones::Zone;

/// Which part of an ability tree a classification question is asking about.
///
/// The distinction is a decision-boundary one, not a convenience one:
///
/// * [`AbilityScope::Unconditional`] answers *"does resolving this ability as
///   already announced produce effect X?"* — the walk a LIVE per-action policy
///   needs. CR 601.2b makes mode selection a distinct step of announcing a
///   spell, so at `CastSpell` time no mode has been chosen yet and a modal
///   branch must not be credited to the cast.
/// * [`AbilityScope::Potential`] answers *"can this card ever produce effect
///   X?"* — the walk DECK-TIME detection needs, where every branch the card
///   could take is in scope.
///
/// A typed scope rather than a `bool` so each call site states which question
/// it is asking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AbilityScope {
    /// `effect` plus the `sub_ability` chain — everything that happens no
    /// matter which branch or mode is taken.
    Unconditional,
    /// The full tree: `Unconditional` plus the `else_ability` branch (CR
    /// 608.2c) and every entry of `mode_abilities` (CR 700.2), recursively.
    Potential,
}

/// Walk an ability tree at `scope`, returning borrowed effect references in
/// traversal order.
///
/// This is the mode-aware authority both deck-time detection and live policy
/// classification share; they differ only in the `scope` they pass, so the two
/// can never drift apart on which branches exist.
pub(crate) fn collect_scoped_effects(
    ability: &AbilityDefinition,
    scope: AbilityScope,
) -> Vec<&Effect> {
    let mut effects: Vec<&Effect> = Vec::new();
    visit_scoped_nodes(ability, scope, &mut |node| effects.push(&*node.effect));
    effects
}

/// Visit every node of an ability tree at `scope`, in traversal order: the
/// node itself, its `sub_ability` chain, then (at [`AbilityScope::Potential`])
/// its `else_ability` branch and each of its `mode_abilities`.
///
/// [`collect_scoped_effects`] is this visitor pushing each node's effect, so
/// the node view and the effect view are one traversal.
pub(crate) fn visit_scoped_nodes<'a>(
    ability: &'a AbilityDefinition,
    scope: AbilityScope,
    visit: &mut impl FnMut(&'a AbilityDefinition),
) {
    visit(ability);
    if let Some(sub) = &ability.sub_ability {
        visit_scoped_nodes(sub, scope, visit);
    }
    if scope == AbilityScope::Unconditional {
        return;
    }
    // CR 608.2c: the "Otherwise, ..." leg is one of two mutually exclusive
    // outcomes, so it is potential rather than unconditional.
    if let Some(other) = &ability.else_ability {
        visit_scoped_nodes(other, scope, visit);
    }
    // CR 700.2: each mode is an alternative the controller may choose.
    for mode in &ability.mode_abilities {
        visit_scoped_nodes(mode, scope, visit);
    }
}

/// Walk `ability.effect` plus each `sub_ability.effect` in turn, returning
/// borrowed references in chain order. Shorthand for
/// [`collect_scoped_effects`] at [`AbilityScope::Unconditional`].
pub(crate) fn collect_chain_effects(ability: &AbilityDefinition) -> Vec<&Effect> {
    collect_scoped_effects(ability, AbilityScope::Unconditional)
}

// ── Flicker authority ───────────────────────────────────────────────────────
//
// The single AI-side authority that recognizes an exile-and-return pairing
// (a flicker) on one ability's own `sub_ability` chain. Every consumer that
// reads an exile leg as removal or as a flicker asks [`ChainNode::flicker_pair`]
// rather than re-deriving the pairing from a flattened effect list.

/// One node of an ability tree, in either of the two tree types a decision can
/// carry: a printed [`AbilityDefinition`] (cast/activation/deck time) or a
/// [`ResolvedAbility`] (a pending cast, trigger or stack entry). One handle
/// lets a single `Vec<ChainNode>` serve every decision kind.
#[derive(Debug, Clone, Copy)]
pub(crate) enum ChainNode<'a> {
    Definition(&'a AbilityDefinition),
    Resolved(&'a ResolvedAbility),
}

impl<'a> ChainNode<'a> {
    /// The node's own effect.
    pub(crate) fn effect(self) -> &'a Effect {
        match self {
            ChainNode::Definition(node) => &node.effect,
            ChainNode::Resolved(node) => &node.effect,
        }
    }

    /// The next node of the same tree along the `sub_ability` link.
    fn next(self) -> Option<ChainNode<'a>> {
        match self {
            ChainNode::Definition(node) => node.sub_ability.as_deref().map(ChainNode::Definition),
            ChainNode::Resolved(node) => node.sub_ability.as_deref().map(ChainNode::Resolved),
        }
    }

    /// The flicker this node's exile leg starts, if any: the exile leg paired
    /// with the first flicker return reached along the node's own
    /// `sub_ability` links. A second exile leg before any return ends the walk
    /// — the nearer exile owns any later return.
    ///
    /// Per-ability by construction: only `sub_ability` links are followed, so
    /// an exile in one ability never pairs with a return in another (CR 607.2a
    /// links Oblivion Ring's ETB exile and LTB return as two abilities, not
    /// one chain).
    pub(crate) fn flicker_pair(self) -> Option<FlickerPair> {
        let exiled = exile_leg_target(self.effect())?;
        let mut node = self.next();
        while let Some(current) = node {
            if exile_leg_target(current.effect()).is_some() {
                return None;
            }
            if let Some(returns) = flicker_return(current.effect()) {
                return Some(FlickerPair {
                    subject: exile_subject(exiled),
                    returns,
                });
            }
            node = current.next();
        }
        None
    }
}

/// Who an exile leg exiles, read from its target filter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ExileSubject {
    /// The ability's own source (`SelfRef`) — a self-blink.
    Source,
    /// A filtered object, with the filter's controller axis (`None` when the
    /// filter imposes no controller — "target creature", "another target
    /// permanent").
    Filtered(Option<ControllerRef>),
}

impl ExileSubject {
    /// The exile can only reach the ability controller's own permanents: the
    /// source itself, or a filter scoped to "you control".
    pub(crate) fn is_own(&self) -> bool {
        match self {
            ExileSubject::Source => true,
            ExileSubject::Filtered(None) => false,
            ExileSubject::Filtered(Some(controller)) => match controller {
                ControllerRef::You => true,
                ControllerRef::Opponent
                | ControllerRef::ScopedPlayer
                | ControllerRef::TargetPlayer
                | ControllerRef::TargetOpponent
                | ControllerRef::ParentTargetController
                | ControllerRef::EventTargetController
                | ControllerRef::ParentTargetOwner
                | ControllerRef::DefendingPlayer
                | ControllerRef::ChosenPlayer { .. }
                | ControllerRef::SourceChosenPlayer
                | ControllerRef::TriggeringPlayer
                | ControllerRef::EnchantedPlayer
                | ControllerRef::ActivePlayer
                | ControllerRef::SpecificPlayer { .. } => false,
            },
        }
    }

    /// The exile can only reach an opponent's permanents (Mystifying Maze's
    /// "attacking creature an opponent controls") — tempo, not a value flicker.
    pub(crate) fn is_opponent_scoped(&self) -> bool {
        match self {
            ExileSubject::Source | ExileSubject::Filtered(None) => false,
            ExileSubject::Filtered(Some(controller)) => match controller {
                ControllerRef::Opponent | ControllerRef::TargetOpponent => true,
                ControllerRef::You
                | ControllerRef::ScopedPlayer
                | ControllerRef::TargetPlayer
                | ControllerRef::ParentTargetController
                | ControllerRef::EventTargetController
                | ControllerRef::ParentTargetOwner
                | ControllerRef::DefendingPlayer
                | ControllerRef::ChosenPlayer { .. }
                | ControllerRef::SourceChosenPlayer
                | ControllerRef::TriggeringPlayer
                | ControllerRef::EnchantedPlayer
                | ControllerRef::ActivePlayer
                | ControllerRef::SpecificPlayer { .. } => false,
            },
        }
    }
}

/// When the exiled card comes back.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ReturnTiming {
    /// In the same resolution, right after the exile.
    Immediate,
    /// CR 603.7: through a delayed triggered ability keyed to the beginning of
    /// a step or phase ("at the beginning of the next end step").
    Scheduled,
}

/// How the exiled card re-enters the battlefield.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ReturnAlteration {
    /// It re-enters as it was printed.
    Unaltered,
    /// CR 712.14a: it re-enters with its back face up.
    Transformed,
    /// CR 122.6: it re-enters with counters the return puts on it.
    WithCounters,
}

/// The return half of a flicker.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct FlickerReturn {
    pub(crate) timing: ReturnTiming,
    /// CR 110.2a: the player the card returns under; `None` is "under its
    /// owner's control" (the engine's encoding — see `ChangeZone.enters_under`).
    pub(crate) enters_under: Option<ControllerRef>,
    pub(crate) alteration: ReturnAlteration,
}

impl FlickerReturn {
    pub(crate) fn is_immediate(&self) -> bool {
        match self.timing {
            ReturnTiming::Immediate => true,
            ReturnTiming::Scheduled => false,
        }
    }

    /// The return itself changes the permanent (transformed, or with
    /// counters), which is the flicker's own payoff.
    pub(crate) fn alters_permanent(&self) -> bool {
        match self.alteration {
            ReturnAlteration::Transformed | ReturnAlteration::WithCounters => true,
            ReturnAlteration::Unaltered => false,
        }
    }
}

/// An exile leg paired with its return.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct FlickerPair {
    pub(crate) subject: ExileSubject,
    pub(crate) returns: FlickerReturn,
}

/// CR 701.13a: to exile an object, move it to the exile zone. The exile leg of
/// a flicker moves a permanent from the battlefield, so an exile from a hand,
/// library or graveyard (Rona, Tolarian Obliterator) is not one.
fn exile_leg_target(effect: &Effect) -> Option<&TargetFilter> {
    match effect {
        Effect::ChangeZone {
            destination: Zone::Exile,
            origin: None | Some(Zone::Battlefield),
            target,
            ..
        } => Some(target),
        _ => None,
    }
}

fn exile_subject(filter: &TargetFilter) -> ExileSubject {
    match filter {
        TargetFilter::SelfRef => ExileSubject::Source,
        other => ExileSubject::Filtered(filter_controller(other)),
    }
}

/// The controller axis a target filter imposes: a `Typed` filter's own
/// controller; for `Or`, the controller every branch agrees on; for `And`, the
/// first conjunct that names one. Every other filter shape imposes no
/// controller (precedent: `features::blink`'s and `anti_self_harm`'s filter
/// readers).
fn filter_controller(filter: &TargetFilter) -> Option<ControllerRef> {
    match filter {
        TargetFilter::Typed(typed) => typed.controller.clone(),
        TargetFilter::Or { filters } => {
            let mut controllers = filters.iter().map(filter_controller);
            let first = controllers.next()??;
            controllers
                .all(|controller| controller.as_ref() == Some(&first))
                .then_some(first)
        }
        TargetFilter::And { filters } => filters.iter().find_map(filter_controller),
        _ => None,
    }
}

/// The return half of a flicker, if `effect` is one: a battlefield return of
/// the exiled object through one of the two anaphors real flicker cards use —
/// the chain's tracked set ("return those cards", Ephemerate, Ghostly Flicker)
/// or the parent target ("return that card", Cloudshift) — either immediately
/// or through a scheduled delayed trigger.
///
/// CR 603.7 + CR 603.7a: a delayed triggered ability created during
/// resolution can return the exiled card at a later time; CR 400.7: the
/// returned card is a new object. A `SelfRef` battlefield return names the
/// source rather than the exiled card, so it is not in the anaphor set.
pub(crate) fn flicker_return(effect: &Effect) -> Option<FlickerReturn> {
    match effect {
        Effect::CreateDelayedTrigger {
            condition, effect, ..
        } if is_scheduled(condition) => collect_chain_effects(effect)
            .into_iter()
            .find_map(immediate_return)
            .map(|returns| FlickerReturn {
                timing: ReturnTiming::Scheduled,
                ..returns
            }),
        _ => immediate_return(effect),
    }
}

fn immediate_return(effect: &Effect) -> Option<FlickerReturn> {
    match effect {
        Effect::ChangeZone {
            destination: Zone::Battlefield,
            target:
                TargetFilter::TrackedSet { .. }
                | TargetFilter::TrackedSetFiltered { .. }
                | TargetFilter::ParentTarget,
            enters_under,
            enter_transformed,
            enter_with_counters,
            ..
        } => Some(FlickerReturn {
            timing: ReturnTiming::Immediate,
            enters_under: enters_under.clone(),
            alteration: return_alteration(*enter_transformed, enter_with_counters),
        }),
        _ => None,
    }
}

/// CR 603.7 + CR 603.7b: a delayed trigger keyed to the beginning of a step or
/// phase triggers the next time that step begins, independent of any other
/// event; an event-keyed delayed trigger ("when … leaves", "when … becomes
/// untapped") triggers only if that event occurs, so its return is contingent.
/// CR 610.3 "until" returns are durations, not flickers.
fn is_scheduled(condition: &DelayedTriggerCondition) -> bool {
    match condition {
        DelayedTriggerCondition::AtNextPhase { .. }
        | DelayedTriggerCondition::AtNextPhaseForPlayer { .. }
        | DelayedTriggerCondition::AtBeginningOfAddedPhase { .. } => true,
        DelayedTriggerCondition::WhenLeavesPlay { .. }
        | DelayedTriggerCondition::WhenDies { .. }
        | DelayedTriggerCondition::WhenLeavesPlayFiltered { .. }
        | DelayedTriggerCondition::WhenEntersBattlefield { .. }
        | DelayedTriggerCondition::WhenDiesOrExiled { .. }
        | DelayedTriggerCondition::WheneverEvent { .. }
        | DelayedTriggerCondition::WhenNextEvent { .. } => false,
    }
}

/// How a return re-enters the permanent. CR 712.14a: a double-faced card put
/// onto the battlefield transformed enters with its back face up (this takes
/// precedence when a return also carries counters); CR 122.6: counters it
/// enters with modify the returned permanent. Either is a change the return
/// itself makes.
fn return_alteration(
    enter_transformed: bool,
    enter_with_counters: &[(CounterType, QuantityExpr)],
) -> ReturnAlteration {
    if enter_transformed {
        ReturnAlteration::Transformed
    } else if !enter_with_counters.is_empty() {
        ReturnAlteration::WithCounters
    } else {
        ReturnAlteration::Unaltered
    }
}

/// True when `node` belongs to an own flicker: it is an exile leg whose paired
/// return makes it a flicker of the source or of a permanent its controller
/// controls, or it is a flicker return. The purity building block of the
/// no-value check (a chain made only of such nodes does nothing but flicker
/// its controller's own permanents).
pub(crate) fn is_own_flicker_node(node: ChainNode<'_>) -> bool {
    node.flicker_pair()
        .is_some_and(|pair| pair.subject.is_own())
        || flicker_return(node.effect()).is_some()
}

#[cfg(test)]
mod tests {
    use engine::game::scenario::{P0, P1};
    use engine::parser::oracle::parse_oracle_text;
    use engine::types::ability::{AbilityKind, TypedFilter};
    use engine::types::identifiers::TrackedSetId;
    use engine::types::zones::EtbTapState;

    use super::*;
    use crate::policies::context::flicker_fixtures as fx;
    use crate::policies::effect_classify::{flicker_target_outcome, FlickerTargetOutcome};

    /// Every ability and trigger execute a card parses to, from its verbatim
    /// Oracle text.
    fn roots(
        name: &str,
        types: &[&str],
        keywords: &[&str],
        oracle: &str,
    ) -> Vec<AbilityDefinition> {
        let types: Vec<String> = types.iter().map(|t| t.to_string()).collect();
        let keywords: Vec<String> = keywords.iter().map(|k| k.to_string()).collect();
        let parsed = parse_oracle_text(oracle, name, &keywords, &types, &[]);
        parsed
            .abilities
            .into_iter()
            .chain(
                parsed
                    .triggers
                    .into_iter()
                    .filter_map(|t| t.execute.map(|e| *e)),
            )
            .collect()
    }

    /// The flicker pair of every node of every root, at `Potential` scope.
    fn pairs(roots: &[AbilityDefinition]) -> Vec<FlickerPair> {
        let mut pairs = Vec::new();
        for root in roots {
            visit_scoped_nodes(root, AbilityScope::Potential, &mut |node| {
                pairs.extend(ChainNode::Definition(node).flicker_pair());
            });
        }
        pairs
    }

    fn only_pair(name: &str, types: &[&str], keywords: &[&str], oracle: &str) -> FlickerPair {
        let pairs = pairs(&roots(name, types, keywords, oracle));
        assert_eq!(
            pairs.len(),
            1,
            "{name}: exactly one flicker pair, got {pairs:?}"
        );
        pairs[0].clone()
    }

    fn change_zone(origin: Option<Zone>, destination: Zone, target: TargetFilter) -> Effect {
        Effect::ChangeZone {
            origin,
            destination,
            target,
            owner_library: false,
            enter_transformed: false,
            enters_under: None,
            enter_tapped: EtbTapState::Unspecified,
            enters_attacking: false,
            up_to: false,
            enter_with_counters: vec![],
            conditional_enter_with_counters: vec![],
            face_down_profile: None,
            enters_modified_if: None,
        }
    }

    fn node(effect: Effect) -> AbilityDefinition {
        AbilityDefinition::new(AbilityKind::Spell, effect)
    }

    fn chain(effects: Vec<Effect>) -> AbilityDefinition {
        effects
            .into_iter()
            .rev()
            .fold(None, |next: Option<AbilityDefinition>, effect| {
                let mut ability = node(effect);
                ability.sub_ability = next.map(Box::new);
                Some(ability)
            })
            .expect("a chain has a node")
    }

    fn own_creature() -> TargetFilter {
        TargetFilter::Typed(TypedFilter::creature().controller(ControllerRef::You))
    }

    fn tracked() -> TargetFilter {
        TargetFilter::TrackedSet {
            id: TrackedSetId(0),
        }
    }

    fn expect(
        pair: &FlickerPair,
        timing: ReturnTiming,
        subject: ExileSubject,
        enters_under: Option<ControllerRef>,
        alteration: ReturnAlteration,
    ) {
        assert_eq!(pair.returns.timing, timing, "{pair:?}");
        assert_eq!(pair.subject, subject, "{pair:?}");
        assert_eq!(pair.returns.enters_under, enters_under, "{pair:?}");
        assert_eq!(pair.returns.alteration, alteration, "{pair:?}");
    }

    // U-A1: immediate and scheduled returns, every subject shape, and the
    // return's alteration.
    #[test]
    fn pairs_immediate_and_scheduled_returns() {
        use ExileSubject::{Filtered, Source};
        use ReturnAlteration::{Transformed, Unaltered, WithCounters};
        use ReturnTiming::{Immediate, Scheduled};
        let you = Some(ControllerRef::You);

        let blink = only_pair(
            "Momentary Blink",
            &["Instant"],
            fx::MOMENTARY_BLINK_KEYWORDS,
            fx::MOMENTARY_BLINK,
        );
        expect(&blink, Immediate, Filtered(you.clone()), None, Unaltered);
        let cloudshift = only_pair("Cloudshift", &["Instant"], &[], fx::CLOUDSHIFT);
        expect(
            &cloudshift,
            Immediate,
            Filtered(you.clone()),
            you.clone(),
            Unaltered,
        );
        let ghostly = only_pair("Ghostly Flicker", &["Instant"], &[], fx::GHOSTLY_FLICKER);
        assert_eq!(ghostly.subject, Filtered(you.clone()));
        let guardian = only_pair(
            "Guardian of Ghirapur",
            &["Creature"],
            &["Flying"],
            fx::GUARDIAN_OF_GHIRAPUR,
        );
        expect(&guardian, Scheduled, Filtered(you.clone()), None, Unaltered);
        let flickerwisp = only_pair("Flickerwisp", &["Creature"], &["Flying"], fx::FLICKERWISP);
        expect(&flickerwisp, Scheduled, Filtered(None), None, Unaltered);
        let mist = only_pair("Turn to Mist", &["Instant"], &[], fx::TURN_TO_MIST);
        expect(&mist, Scheduled, Filtered(None), None, Unaltered);
        // The tapped return is not an alteration (it is not a gain).
        let displacer = only_pair(
            "Eldrazi Displacer",
            &["Creature"],
            fx::ELDRAZI_DISPLACER_KEYWORDS,
            fx::ELDRAZI_DISPLACER,
        );
        expect(&displacer, Immediate, Filtered(None), None, Unaltered);
        let maze = only_pair("Mystifying Maze", &["Land"], &[], fx::MYSTIFYING_MAZE);
        expect(
            &maze,
            Scheduled,
            Filtered(Some(ControllerRef::Opponent)),
            None,
            Unaltered,
        );
        let fleeting = only_pair("Fleeting Spirit", &["Creature"], &[], fx::FLEETING_SPIRIT);
        expect(&fleeting, Scheduled, Source, None, Unaltered);
        let aetherling = only_pair("Aetherling", &["Creature"], &[], fx::AETHERLING);
        expect(&aetherling, Scheduled, Source, None, Unaltered);
        let huatli = only_pair("Huatli, Poet of Unity", &["Creature"], &[], fx::HUATLI);
        expect(&huatli, Immediate, Source, None, Transformed);
        let daydream = only_pair(
            "Daydream",
            &["Sorcery"],
            fx::DAYDREAM_KEYWORDS,
            fx::DAYDREAM,
        );
        expect(&daydream, Immediate, Filtered(you), None, WithCounters);

        // Flickering Spirit's return names the source (`SelfRef`), not the
        // exiled card: outside the anaphor set.
        assert!(pairs(&roots(
            "Flickering Spirit",
            &["Creature"],
            &["Flying"],
            fx::FLICKERING_SPIRIT
        ))
        .is_empty());
    }

    // U-A2: lone exiles, reanimation, non-anaphoric and non-battlefield
    // shapes, contingent delayed returns and reversed order are not flickers.
    #[test]
    fn non_pairs_are_plain() {
        for (name, types, keywords, oracle) in [
            (
                "Swords to Plowshares",
                "Instant",
                &[][..],
                fx::SWORDS_TO_PLOWSHARES,
            ),
            ("Banisher Priest", "Creature", &[][..], fx::BANISHER_PRIEST),
            ("Zombify", "Sorcery", &[][..], fx::ZOMBIFY),
            (
                "Rona, Tolarian Obliterator",
                "Creature",
                &["Trample"][..],
                fx::RONA,
            ),
            // CR 603.7b: "when this artifact leaves the battlefield or becomes
            // untapped" is a contingent return.
            ("Tawnos's Coffin", "Artifact", &[][..], fx::TAWNOS_COFFIN),
        ] {
            assert!(
                pairs(&roots(name, &[types], keywords, oracle)).is_empty(),
                "{name} must not pair"
            );
        }

        let typed_return = chain(vec![
            change_zone(None, Zone::Exile, own_creature()),
            change_zone(
                None,
                Zone::Battlefield,
                TargetFilter::Typed(TypedFilter::creature()),
            ),
        ]);
        let contingent = chain(vec![
            change_zone(None, Zone::Exile, own_creature()),
            Effect::CreateDelayedTrigger {
                condition: DelayedTriggerCondition::WhenLeavesPlay {
                    object_id: engine::types::identifiers::ObjectId(1),
                },
                effect: Box::new(node(change_zone(
                    Some(Zone::Exile),
                    Zone::Battlefield,
                    TargetFilter::ParentTarget,
                ))),
                uses_tracked_set: false,
            },
        ]);
        let reversed = chain(vec![
            change_zone(None, Zone::Battlefield, tracked()),
            change_zone(None, Zone::Exile, own_creature()),
        ]);
        let hand_exile = chain(vec![
            change_zone(Some(Zone::Hand), Zone::Exile, own_creature()),
            change_zone(None, Zone::Battlefield, TargetFilter::ParentTarget),
        ]);
        assert!(pairs(&[typed_return, contingent, reversed, hand_exile]).is_empty());
    }

    // U-A3: the nearer exile owns the return.
    #[test]
    fn two_exiles_pair_only_the_nearer() {
        let ability = chain(vec![
            change_zone(None, Zone::Exile, own_creature()),
            change_zone(None, Zone::Exile, own_creature()),
            change_zone(None, Zone::Battlefield, tracked()),
        ]);
        let first = ChainNode::Definition(&ability);
        let second = ChainNode::Definition(ability.sub_ability.as_deref().unwrap());
        assert!(first.flicker_pair().is_none());
        assert!(second.flicker_pair().is_some());
    }

    // U-A4: Oblivion Ring's ETB exile and LTB return are two abilities.
    #[test]
    fn pairing_never_crosses_abilities() {
        let ring = roots("Oblivion Ring", &["Enchantment"], &[], fx::OBLIVION_RING);
        assert_eq!(ring.len(), 2, "ETB exile and LTB return");
        assert!(pairs(&ring).is_empty());
    }

    // U-A5: Ghostly Flicker's one return pairs with both exiled targets.
    #[test]
    fn ghostly_flicker_pairs_both_targets() {
        let pair = only_pair("Ghostly Flicker", &["Instant"], &[], fx::GHOSTLY_FLICKER);
        let mut scenario = fx::scenario();
        let giant = fx::giant(&mut scenario, P0);
        let bears = fx::bears(&mut scenario, P0);
        let token = fx::token_body(&mut scenario, P0);
        let _ = P1;
        let mut runner = scenario.build();
        fx::make_token(runner.state_mut(), token);
        let state = runner.state();
        assert_eq!(
            flicker_target_outcome(state, &pair, giant, P0),
            Some(FlickerTargetOutcome::Reenters)
        );
        assert_eq!(
            flicker_target_outcome(state, &pair, bears, P0),
            Some(FlickerTargetOutcome::Reenters)
        );
        assert_eq!(
            flicker_target_outcome(state, &pair, token, P0),
            Some(FlickerTargetOutcome::Ceases)
        );
    }

    fn nodes_of(root: &AbilityDefinition) -> Vec<&AbilityDefinition> {
        let mut nodes = Vec::new();
        visit_scoped_nodes(root, AbilityScope::Potential, &mut |node| nodes.push(node));
        nodes
    }

    fn all_own_flicker_nodes(roots: &[AbilityDefinition]) -> bool {
        roots
            .iter()
            .flat_map(nodes_of)
            .all(|node| is_own_flicker_node(ChainNode::Definition(node)))
    }

    // U-A6: the purity building block of the no-value check.
    #[test]
    fn own_flicker_node_purity() {
        for (name, keywords, oracle) in [
            (
                "Momentary Blink",
                fx::MOMENTARY_BLINK_KEYWORDS,
                fx::MOMENTARY_BLINK,
            ),
            ("Cloudshift", &[][..], fx::CLOUDSHIFT),
            ("Ghostly Flicker", &[][..], fx::GHOSTLY_FLICKER),
        ] {
            assert!(
                all_own_flicker_nodes(&roots(name, &["Instant"], keywords, oracle)),
                "{name}"
            );
        }
        // The self-blink activations (the exile activation is the ability whose
        // root exiles).
        for (name, oracle) in [
            ("Aetherling", fx::AETHERLING),
            ("Huatli, Poet of Unity", fx::HUATLI),
            ("Fleeting Spirit", fx::FLEETING_SPIRIT),
        ] {
            let blink: Vec<AbilityDefinition> = roots(name, &["Creature"], &[], oracle)
                .into_iter()
                .filter(|root| exile_leg_target(&root.effect).is_some())
                .collect();
            assert_eq!(blink.len(), 1, "{name}: one self-blink activation");
            assert!(all_own_flicker_nodes(&blink), "{name}");
        }

        let stratagem = roots(
            "Illusionist's Stratagem",
            &["Instant"],
            &[],
            fx::ILLUSIONISTS_STRATAGEM,
        );
        let draw = stratagem
            .iter()
            .flat_map(nodes_of)
            .find(|node| matches!(*node.effect, Effect::Draw { .. }))
            .expect("Stratagem draws");
        assert!(!is_own_flicker_node(ChainNode::Definition(draw)));

        let settle = roots(
            "Settle Beyond Reality",
            &["Sorcery"],
            &[],
            fx::SETTLE_BEYOND_REALITY,
        );
        let opponent_exile = settle
            .iter()
            .flat_map(nodes_of)
            .find(|node| {
                exile_leg_target(&node.effect).is_some()
                    && ChainNode::Definition(node).flicker_pair().is_none()
            })
            .expect("Settle's removal mode");
        assert!(!is_own_flicker_node(ChainNode::Definition(opponent_exile)));

        let mist = roots("Turn to Mist", &["Instant"], &[], fx::TURN_TO_MIST);
        assert!(!is_own_flicker_node(ChainNode::Definition(&mist[0])));

        let kaya = roots(
            "Kaya, Ghost Assassin",
            &["Planeswalker"],
            &[],
            fx::KAYA_GHOST_ASSASSIN,
        );
        let lose_life = kaya
            .iter()
            .flat_map(nodes_of)
            .find(|node| matches!(*node.effect, Effect::LoseLife { .. }))
            .expect("Kaya's [0] loses life");
        assert!(!is_own_flicker_node(ChainNode::Definition(lose_life)));
    }
}
