//! Blink / flicker (ETB-value reuse) feature — structural detection over a
//! deck's typed AST.
//!
//! Parser AST verification — VERIFIED (no parser remediation required; every
//! axis classifies from the existing typed AST, never by card name):
//! - Flicker enabler: an ability/trigger chain whose exile leg moves a
//!   permanent from the battlefield (CR 701.13a) and is paired, along that
//!   leg's own `sub_ability` links, with a return of the exiled card to the
//!   battlefield — immediate (Ephemerate:
//!   `ChangeZone { destination: Exile, target: Typed(Creature, controller: You) }`
//!   followed by `ChangeZone { destination: Battlefield, target: TrackedSet }`)
//!   or through a delayed trigger at the beginning of the next end step
//!   (CR 603.7: Guardian of Ghirapur, Flickerwisp, Fleeting Spirit). The
//!   pairing is the single flicker authority
//!   [`crate::ability_chain::ChainNode::flicker_pair`]. The return names the
//!   exiled card through the tracked-set / parent-target anaphor, which is
//!   what distinguishes a flicker-return from a graveyard reanimation and from
//!   a one-way removal exile. Deck-time detection walks every branch a card
//!   can take (`AbilityScope::Potential`), so a modal blink mode (CR 700.2:
//!   Kykar, Zephyr Awakener) counts; an exile from a hand, library or
//!   graveyard (Rona, Tolarian Obliterator) does not.
//! - ETB-value payoff: a creature whose `TriggerMode::ChangesZone` trigger fires
//!   on itself or a friendly creature entering the battlefield (CR 603.6a) and
//!   whose executed chain produces card-advantage / board / removal value worth
//!   re-triggering — Mulldrifter parses as `mode: ChangesZone`,
//!   `valid_card: SelfRef`, `destination: Battlefield`, executing `Draw`.
//!
//! Why this is not redundant with existing handling: `policies/etb_value.rs`
//! scores an ETB trigger's value at *cast time only* (one-shot), with no notion
//! of re-triggering it; `aristocrats` keys on sacrifice/death and `tokens_wide`
//! on going wide — none recognizes a deck whose plan is "flicker a creature to
//! re-use its ETB." This axis fills that gap; the companion `BlinkPayoffPolicy`
//! is payoff-gated so non-blink decks (and the general `etb_value` scoring) are
//! unaffected.

use engine::game::game_object::GameObject;
use engine::game::DeckEntry;
use engine::types::ability::{
    AbilityDefinition, ControllerRef, Effect, TargetFilter, TriggerDefinition, TypeFilter,
};
use engine::types::card::CardFace;
use engine::types::card_type::CoreType;
use engine::types::triggers::TriggerMode;
use engine::types::zones::Zone;

use crate::ability_chain::{collect_chain_effects, visit_scoped_nodes, AbilityScope, ChainNode};
use crate::features::commitment;

/// Commitment floor below which `BlinkPayoffPolicy` opts out. Matches the
/// reanimator / equipment / enchantments payoff-axis convention.
pub const COMMITMENT_FLOOR: f32 = 0.30;

/// Per-60-nonland flicker-enabler density at which the flicker pillar saturates.
/// Dedicated blink shells run ~6–10 flicker enablers (Ephemerate, Cloudshift,
/// Ghostly Flicker, Soulherder, Restoration Angel) per 60 nonland; the divisor
/// is set so a single incidental flicker spell stays below the floor.
const FLICKER_FULL_DENSITY: f32 = 7.0;

/// Per-60-nonland ETB-payoff density at which the payoff pillar saturates. A
/// committed blink deck runs ~12–18 value-ETB creatures (Mulldrifter, Elvish
/// Visionary, Wall of Omens, Solemn Simulacrum, Flametongue Kavu, Ravenous
/// Chupacabra) per 60 nonland.
const ETB_PAYOFF_FULL_DENSITY: f32 = 12.0;

/// Per-deck blink / flicker classification.
///
/// Populated once per game from `DeckEntry` data. Detection is structural over
/// `CardFace.{abilities,triggers,card_type}` — never by card name. The companion
/// `BlinkPayoffPolicy` consumes this to value deploying flicker enablers and
/// casting value-ETB creatures when the deck is blink-committed.
#[derive(Debug, Clone, Default)]
pub struct BlinkFeature {
    /// Cards with a chain that exiles a friendly or unscoped permanent from
    /// the battlefield and returns it — immediately, or through a delayed
    /// trigger at the beginning of a later step (CR 603.7) — in any branch or
    /// mode the card can take (CR 700.2). The flicker engine.
    pub flicker_count: u32,
    /// Value-ETB creatures — a `ChangesZone`→battlefield self/friendly trigger
    /// producing card-advantage / board / removal value (CR 603.6a). The payoff
    /// being re-triggered. Without flicker there is nothing to re-trigger, and
    /// without payoffs flicker has nothing worth re-using, so both are required.
    pub etb_payoff_count: u32,
    /// `0.0..=1.0` — how central the blink plan is to this deck. Requires both
    /// flicker density and ETB-payoff density; missing either collapses to
    /// inert. Consumed by `BlinkPayoffPolicy::activation` as the scaling knob.
    pub commitment: f32,
}

/// Structural detection — walks each `DeckEntry`'s `CardFace` AST and counts
/// flicker enablers and value-ETB payoffs.
pub fn detect(deck: &[DeckEntry]) -> BlinkFeature {
    if deck.is_empty() {
        return BlinkFeature::default();
    }

    let mut flicker_count = 0u32;
    let mut etb_payoff_count = 0u32;
    let mut total_nonland = 0u32;

    for entry in deck {
        let face = &entry.card;
        if !face.card_type.core_types.contains(&CoreType::Land) {
            total_nonland = total_nonland.saturating_add(entry.count);
        }
        if is_flicker_enabler(face) {
            flicker_count = flicker_count.saturating_add(entry.count);
        }
        if is_etb_payoff(face) {
            etb_payoff_count = etb_payoff_count.saturating_add(entry.count);
        }
    }

    let commitment = blink_commitment(flicker_count, etb_payoff_count, total_nonland);

    BlinkFeature {
        flicker_count,
        etb_payoff_count,
        commitment,
    }
}

/// Geometric-mean commitment over the two required pillars (flicker density and
/// ETB-payoff density).
///
/// A blink deck needs BOTH a way to flicker AND ETBs worth re-triggering;
/// missing either pillar means it is not the blink plan, so commitment collapses
/// to `0.0` (which keeps the policy opted out for decks that merely run one
/// incidental flicker spell or a couple of value creatures).
///
/// Calibration — WX Blink (≈38 nonland: 8 flicker enablers, 14 value-ETB
/// creatures): flicker density ≈12.6 and ETB density ≈22.1 both saturate their
/// pillars → geometric mean 1.0, well above the floor.
///
/// Anti-calibration — one incidental flicker spell + two value-ETB creatures
/// (≈36 nonland) gives flicker density ≈1.67 and ETB density ≈3.33 → geometric
/// mean ≈0.26, below the floor, so the policy stays inert.
fn blink_commitment(flicker_count: u32, etb_payoff_count: u32, total_nonland: u32) -> f32 {
    if flicker_count == 0 || etb_payoff_count == 0 || total_nonland == 0 {
        return 0.0;
    }

    let flicker =
        (commitment::density_per_60(flicker_count, total_nonland) / FLICKER_FULL_DENSITY).min(1.0);
    let payoff = (commitment::density_per_60(etb_payoff_count, total_nonland)
        / ETB_PAYOFF_FULL_DENSITY)
        .min(1.0);

    commitment::geometric_mean(&[flicker, payoff]).min(1.0)
}

/// True if this face is a flicker enabler — at least one of its ability or
/// trigger chains pairs an exile of a friendly or unscoped permanent with a
/// return of the exiled card (CR 603.7), in any branch or mode the card can
/// take.
///
/// Deck-time detection asks "can this card ever flicker?", which is
/// [`AbilityScope::Potential`]. Pairing is per-ability by construction, so two
/// independent abilities (one that exiles and one that puts something onto
/// the battlefield) cannot combine into a false positive.
pub fn is_flicker_enabler(face: &CardFace) -> bool {
    face.abilities
        .iter()
        .any(|ability| ability_is_flicker_engine(ability, AbilityScope::Potential))
        || face.triggers.iter().any(|trigger| {
            trigger
                .execute
                .as_deref()
                .is_some_and(|execute| ability_is_flicker_engine(execute, AbilityScope::Potential))
        })
}

/// True if this face is a value-ETB payoff: a creature with a self/friendly
/// `ChangesZone`→battlefield trigger producing card-advantage / board / removal
/// value worth re-triggering. CR 603.6a.
pub fn is_etb_payoff(face: &CardFace) -> bool {
    face_is_creature(face) && face.triggers.iter().any(trigger_is_value_etb)
}

/// Parts predicate — true if some node of `ability`, walked at `scope`, starts
/// a flicker (an exile leg paired with its return) whose subject is not
/// opponent-scoped. An opponent-scoped exile is tempo, not a value flicker.
/// Shared by deck-time detection ([`is_flicker_enabler`], `Potential`) and the
/// live `BlinkPayoffPolicy` (`Unconditional`), which differ only in the scope
/// they pass.
pub(crate) fn ability_is_flicker_engine(ability: &AbilityDefinition, scope: AbilityScope) -> bool {
    let mut found = false;
    visit_scoped_nodes(ability, scope, &mut |node| {
        found = found
            || ChainNode::Definition(node)
                .flicker_pair()
                .is_some_and(|pair| !pair.subject.is_opponent_scoped());
    });
    found
}

/// CR 603.6a: true if `object` has a self/friendly value ETB — the payoff a
/// flicker re-uses.
pub(crate) fn object_has_value_etb(object: &GameObject) -> bool {
    object
        .trigger_definitions
        .iter_unchecked()
        .map(|entry| &entry.definition)
        .any(trigger_is_value_etb)
}

/// Single authority — true if this trigger is a self/friendly value ETB. Shared
/// by the detector and the live policy (which classifies a live `GameObject`'s
/// `trigger_definitions` without reconstructing a `CardFace`).
pub(crate) fn trigger_is_value_etb(trigger: &TriggerDefinition) -> bool {
    if trigger.mode != TriggerMode::ChangesZone {
        return false;
    }
    // CR 603.6a: "enters the battlefield". A trigger whose origin is the
    // battlefield is a "leaves" trigger masquerading as ChangesZone — exclude it.
    if trigger.destination != Some(Zone::Battlefield)
        || matches!(trigger.origin, Some(Zone::Battlefield))
    {
        return false;
    }
    let Some(valid_card) = trigger.valid_card.as_ref() else {
        return false;
    };
    if !etb_filter_is_self_or_friendly_creature(valid_card) {
        return false;
    }
    trigger.execute.as_deref().is_some_and(|execute| {
        collect_chain_effects(execute)
            .iter()
            .copied()
            .any(effect_is_etb_value)
    })
}

/// CR 603.6a: the trigger fires on the source itself entering (`SelfRef`, the
/// common "When ~ enters" self-ETB) or on a friendly creature entering (an
/// "whenever a creature you control enters" engine). An opponent-scoped or
/// non-creature filter is not a creature-ETB-value payoff.
///
/// Separated into two orthogonal checks so that compound filters like
/// "friendly white creature" (`And { [Typed(Creature), Typed(White)] }`) are
/// accepted: the creature check uses `.any()` over `And` conjuncts (at least
/// one conjunct must name a creature type), while the opponent-scope check
/// delegates to `target_is_not_opponent_scoped` which uses `.all()` (no
/// conjunct may be opponent-scoped).
fn etb_filter_is_self_or_friendly_creature(filter: &TargetFilter) -> bool {
    target_is_not_opponent_scoped(filter) && filter_contains_creature(filter)
}

fn filter_contains_creature(filter: &TargetFilter) -> bool {
    match filter {
        TargetFilter::SelfRef => true,
        TargetFilter::Typed(typed) => typed.type_filters.iter().any(type_filter_is_creature),
        TargetFilter::Or { filters } => filters.iter().any(filter_contains_creature),
        TargetFilter::And { filters } => filters.iter().any(filter_contains_creature),
        _ => false,
    }
}

fn type_filter_is_creature(tf: &TypeFilter) -> bool {
    match tf {
        TypeFilter::Creature => true,
        TypeFilter::AnyOf(inner) => inner.iter().any(type_filter_is_creature),
        _ => false,
    }
}

/// Unwrap an ETB trigger's `valid_card` filter and report whether it is NOT
/// opponent-scoped (a friendly or unscoped creature whose entering the deck
/// re-uses). `And` rejects if any conjunct is opponent-scoped.
fn target_is_not_opponent_scoped(filter: &TargetFilter) -> bool {
    match filter {
        TargetFilter::Typed(typed) => !matches!(typed.controller, Some(ControllerRef::Opponent)),
        TargetFilter::Or { filters } => filters.iter().any(target_is_not_opponent_scoped),
        TargetFilter::And { filters } => filters.iter().all(target_is_not_opponent_scoped),
        // SelfRef / Any / unscoped references are friendly-usable.
        TargetFilter::SelfRef | TargetFilter::Any => true,
        _ => false,
    }
}

/// The curated set of ETB effects worth re-triggering via flicker — the value a
/// blink deck is built to re-use. Each covers a class of canonical blink
/// targets:
/// - `Draw` — card advantage (Mulldrifter, Elvish Visionary)
/// - `Token` — board presence (Cloudgoat Ranger)
/// - `DealDamage` — removal / reach (Flametongue Kavu)
/// - `Destroy` — removal (Ravenous Chupacabra, Shriekmaw)
/// - `SearchLibrary` — tutor / ramp (Solemn Simulacrum, Ranger of Eos)
/// - `Bounce` — tempo (Man-o'-War)
fn effect_is_etb_value(effect: &Effect) -> bool {
    matches!(
        effect,
        Effect::Draw { .. }
            | Effect::Token { .. }
            | Effect::DealDamage { .. }
            | Effect::Destroy { .. }
            | Effect::SearchLibrary { .. }
            | Effect::Bounce { .. }
    )
}

/// CR 302.1: a creature card — the body a blink deck flickers to re-use its ETB.
fn face_is_creature(face: &CardFace) -> bool {
    face.card_type.core_types.contains(&CoreType::Creature)
}
