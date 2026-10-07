//! CR 107.1 + CR 614.1c: an enters-with replacement whose leading amount is
//! dynamic ("X", "twice X", "that many") and is placed once FOR EACH object of
//! a "for each" count — alone or conjoined with a further "and … for each"
//! placement — needs a product of two game-state values that `QuantityExpr`
//! cannot encode. The whole replacement is declined, so the card is reported
//! UNSUPPORTED through the public coverage authority (`card_face_gaps`); it
//! never becomes a supported count that drops the leading amount (X=3 over two
//! other red and one other green creature must be 3×2+1=7, never 2+1). Each
//! case is paired with a fixed leading amount over the same populations, which
//! stays represented with no gap and places its counters at runtime.
//!
//! Under "plus an additional … for each" the leading amount is an addend, not
//! a factor, so a dynamic base composes faithfully as a sum and stays
//! supported. Synthetic class cards (no printed card yet), except Sheriff of
//! Safe Passage (verbatim Oracle text).

use engine::game::coverage::card_face_gaps;
use engine::game::scenario::{GameScenario, P0, P1};
use engine::parser::parse_oracle_text;
use engine::types::ability::{Effect, QuantityExpr, QuantityRef};
use engine::types::card::CardFace;
use engine::types::counter::CounterType;
use engine::types::identifiers::ObjectId;
use engine::types::mana::{ManaColor, ManaCost, ManaCostShard, ManaType, ManaUnit};
use engine::types::phase::Phase;
use engine::types::zones::Zone;

const DYNAMIC_COMPOUND: &str = "This creature enters with X +1/+1 counters on it for each other \
     red creature you control and a +1/+1 counter on it for each other green creature you \
     control.";
const FIXED_COMPOUND: &str = "This creature enters with two +1/+1 counters on it for each other \
     red creature you control and a +1/+1 counter on it for each other green creature you \
     control.";
const DYNAMIC_BASE_PLUS_ADDITIONAL: &str = "This creature enters with X +1/+1 counters on it plus \
     an additional +1/+1 counter on it for each other creature you control.";

fn face(name: &str, oracle: &str) -> CardFace {
    let parsed = parse_oracle_text(oracle, name, &[], &["Creature".to_string()], &[]);
    CardFace {
        name: name.to_string(),
        oracle_text: Some(oracle.to_string()),
        abilities: parsed.abilities,
        triggers: parsed.triggers,
        static_abilities: parsed.statics,
        replacements: parsed.replacements,
        keywords: parsed.extracted_keywords,
        ..Default::default()
    }
}

/// The counts of every enters-with `PutCounter` the face's replacements place.
fn enters_with_counts(face: &CardFace) -> Vec<QuantityExpr> {
    face.replacements
        .iter()
        .filter_map(|def| def.execute.as_deref())
        .filter_map(|execute| match execute.effect.as_ref() {
            Effect::PutCounter { count, .. } => Some(count.clone()),
            _ => None,
        })
        .collect()
}

/// Assert the line is unsupported: no counter placement is claimed, the whole
/// line is recorded as an unimplemented replacement structure, and that is the
/// card's coverage gap.
fn assert_unsupported(text: &str) {
    // The parser normalizes the self-reference to `~`; the rest is verbatim.
    let body = text
        .strip_prefix("This creature ")
        .expect("synthetic lines name the creature");
    let face = face("Dynamic Hellion", text);
    assert_eq!(
        enters_with_counts(&face),
        Vec::<QuantityExpr>::new(),
        "{text:?}: no partial count may be published"
    );
    let unimplemented: Vec<&str> = face
        .abilities
        .iter()
        .filter_map(|def| def.effect.unimplemented_description())
        .collect();
    assert!(
        matches!(unimplemented.as_slice(), [line] if line.ends_with(body)),
        "{text:?}: the whole line must be the unimplemented node, got {:?}",
        face.abilities
    );
    assert_eq!(
        card_face_gaps(&face),
        vec!["Effect:replacement_structure".to_string()],
        "{text:?}"
    );
}

/// CR 107.1 + CR 614.1c: X counters for each other red creature and a counter
/// for each other green creature is unsupported, never the sum of the two
/// populations; two counters for each other red creature over the same
/// populations is the represented control.
#[test]
fn dynamic_amount_per_each_with_a_further_conjunct_is_unsupported() {
    let control = face("Fixed Hellion", FIXED_COMPOUND);
    let counts = enters_with_counts(&control);
    assert!(
        matches!(
            counts.as_slice(),
            [QuantityExpr::Sum { exprs }]
                if matches!(
                    exprs.as_slice(),
                    [
                        QuantityExpr::Multiply { factor: 2, inner },
                        QuantityExpr::Ref { .. },
                    ] if matches!(inner.as_ref(), QuantityExpr::Ref { .. })
                )
        ),
        "reach guard: {counts:?}"
    );
    assert_eq!(card_face_gaps(&control), Vec::<String>::new());

    assert_unsupported(DYNAMIC_COMPOUND);
    assert_unsupported(
        "This creature enters with twice X +1/+1 counters on it for each other red creature \
         you control and a +1/+1 counter on it for each other green creature you control.",
    );
}

/// CR 107.1 + CR 614.1c: a single per-each clause scaled by a dynamic amount
/// is unsupported, never the bare population count; a fixed amount is the
/// represented control.
#[test]
fn dynamic_amount_for_each_single_clause_is_unsupported() {
    let control = face(
        "Fixed Hellion",
        "This creature enters with two +1/+1 counters on it for each other creature you \
         control.",
    );
    let counts = enters_with_counts(&control);
    assert!(
        matches!(
            counts.as_slice(),
            [QuantityExpr::Multiply { factor: 2, inner }]
                if matches!(inner.as_ref(), QuantityExpr::Ref { .. })
        ),
        "reach guard: {counts:?}"
    );
    assert_eq!(card_face_gaps(&control), Vec::<String>::new());

    assert_unsupported(
        "This creature enters with X +1/+1 counters on it for each other creature you control.",
    );
}

/// CR 107.1 + CR 122.1: a dynamic base under "plus an additional … for each"
/// is an addend, so it composes as `X + count` and stays supported — never the
/// bare base; Sheriff of Safe Passage's fixed base (verbatim) is the control.
#[test]
fn dynamic_base_plus_additional_for_each_sums_base_and_bonus() {
    let sheriff = face(
        "Sheriff of Safe Passage",
        "This creature enters with a +1/+1 counter on it plus an additional +1/+1 counter on \
         it for each other creature you control.",
    );
    let counts = enters_with_counts(&sheriff);
    assert!(
        matches!(
            counts.as_slice(),
            [QuantityExpr::Offset { offset: 1, inner }]
                if matches!(inner.as_ref(), QuantityExpr::Ref { .. })
        ),
        "reach guard: {counts:?}"
    );
    assert_eq!(card_face_gaps(&sheriff), Vec::<String>::new());

    let dynamic = face("Dynamic Hellion", DYNAMIC_BASE_PLUS_ADDITIONAL);
    let counts = enters_with_counts(&dynamic);
    assert!(
        matches!(
            counts.as_slice(),
            [QuantityExpr::Sum { exprs }]
                if matches!(
                    exprs.as_slice(),
                    [
                        QuantityExpr::Ref { qty: QuantityRef::CostXPaid },
                        QuantityExpr::Ref { qty: QuantityRef::ObjectCount { .. } },
                    ]
                )
        ),
        "{counts:?}"
    );
    assert_eq!(card_face_gaps(&dynamic), Vec::<String>::new());
}

/// CR 107.3 + CR 614.1c: an X defined by a trailing "where X is" clause is
/// still a dynamic amount. Scaled per object it needs a product; under "plus an
/// additional … for each" the X-binding override would replace the composed
/// count wholesale. Both decline the whole replacement — never the bare
/// "where X" quantity with the per-each count dropped.
#[test]
fn where_x_amount_with_a_per_each_tail_is_unsupported() {
    assert_unsupported(
        "This creature enters with X +1/+1 counters on it for each other creature you \
         control, where X is the number of cards in your hand.",
    );
    assert_unsupported(
        "This creature enters with X +1/+1 counters on it plus an additional +1/+1 counter on \
         it for each other creature you control, where X is the number of cards in your hand.",
    );
}

/// CR 614.1c + CR 122.1: the fixed control places two counters per other red
/// creature and one per other green creature: two red allies and one green ally
/// give 2×2+1=5; the opponent's red creature counts for neither.
#[test]
fn fixed_amount_per_each_with_a_further_conjunct_places_its_counters() {
    let mut scenario = GameScenario::new();
    scenario.at_phase(Phase::PreCombatMain);
    for name in ["Red Ally A", "Red Ally B"] {
        scenario
            .add_creature(P0, name, 2, 2)
            .with_color(vec![ManaColor::Red]);
    }
    scenario
        .add_creature(P0, "Green Ally", 2, 2)
        .with_color(vec![ManaColor::Green]);
    scenario
        .add_creature(P1, "Red Foe", 2, 2)
        .with_color(vec![ManaColor::Red]);
    let hellion = scenario
        .add_creature_to_hand_from_oracle(P0, "Fixed Hellion", 0, 0, FIXED_COMPOUND)
        .with_mana_cost(ManaCost::Cost {
            generic: 2,
            shards: vec![],
        })
        .id();
    scenario.with_mana_pool(
        P0,
        (0..2)
            .map(|_| ManaUnit::new(ManaType::Colorless, ObjectId(0), false, vec![]))
            .collect(),
    );
    let mut runner = scenario.build();
    let outcome = runner.cast(hellion).resolve();
    outcome.assert_zone(&[hellion], Zone::Battlefield);
    assert_eq!(
        runner.state().objects[&hellion]
            .counters
            .get(&CounterType::Plus1Plus1),
        Some(&5)
    );
}

/// CR 107.3m + CR 122.1: cast for X=4 beside two other creatures you control
/// (and one opponent's), the dynamic base plus one counter per other creature
/// places 4+2=6 — neither the base alone (4), the bonus alone (2), nor the
/// product (8).
#[test]
fn dynamic_base_plus_additional_for_each_places_base_and_bonus() {
    let mut scenario = GameScenario::new();
    scenario.at_phase(Phase::PreCombatMain);
    for name in ["Ally A", "Ally B"] {
        scenario.add_creature(P0, name, 2, 2);
    }
    scenario.add_creature(P1, "Foe", 2, 2);
    let hellion = scenario
        .add_creature_to_hand_from_oracle(P0, "Dynamic Hellion", 0, 0, DYNAMIC_BASE_PLUS_ADDITIONAL)
        .with_mana_cost(ManaCost::Cost {
            generic: 0,
            shards: vec![ManaCostShard::X, ManaCostShard::Green],
        })
        .id();
    scenario.with_mana_pool(
        P0,
        std::iter::once(ManaUnit::new(ManaType::Green, ObjectId(0), false, vec![]))
            .chain((0..4).map(|_| ManaUnit::new(ManaType::Colorless, ObjectId(0), false, vec![])))
            .collect(),
    );
    let mut runner = scenario.build();
    let outcome = runner.cast(hellion).x(4).resolve();
    outcome.assert_zone(&[hellion], Zone::Battlefield);
    assert_eq!(
        runner.state().objects[&hellion]
            .counters
            .get(&CounterType::Plus1Plus1),
        Some(&6)
    );
}
