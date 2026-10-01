//! Public combat regressions; Oracle verified against MTGJSON AtomicCards.
use engine::game::combat::AttackTarget;
use engine::game::scenario::{GameRunner, GameScenario, P0, P1};
use engine::types::actions::GameAction;
use engine::types::events::GameEvent;
use engine::types::game_state::{StackEntryKind, WaitingFor};
use engine::types::identifiers::ObjectId;
use engine::types::keywords::Keyword;
use engine::types::phase::Phase;
use engine::types::triggers::TriggerMode;

const HIGH_GROUND: &str = "Each creature you control can block an additional creature each combat.";
const SUSTAINER: &str = "Flying\nWhenever this creature blocks, it gets +0/+2 until end of turn.";
const WALL: &str = "Defender\nWhenever this creature blocks a creature, that creature doesn't untap during its controller's next untap step.";
const SAWJACK: &str = "Reach (This creature can block creatures with flying.)\nWhenever this creature blocks a creature with flying, this creature gets +2/+0 until end of turn.";
const CADETS: &str = "Whenever this creature blocks or becomes blocked, target opponent gains control of it. (This removes this creature from combat.)";

fn priority_to(runner: &mut GameRunner, blockers: bool) {
    for _ in 0..32 {
        if matches!(
            runner.state().waiting_for,
            WaitingFor::DeclareBlockers { .. }
        ) && blockers
            || matches!(
                runner.state().waiting_for,
                WaitingFor::DeclareAttackers { .. }
            ) && !blockers
        {
            return;
        }
        assert!(matches!(
            runner.state().waiting_for,
            WaitingFor::Priority { .. }
        ));
        runner.act(GameAction::PassPriority).unwrap();
    }
    panic!("combat prompt not reached");
}

fn declare(runner: &mut GameRunner, attackers: &[ObjectId], sources: &[ObjectId]) {
    priority_to(runner, false);
    runner
        .act(GameAction::DeclareAttackers {
            attacks: attackers
                .iter()
                .map(|id| (*id, AttackTarget::Player(P1)))
                .collect(),
            bands: vec![],
        })
        .unwrap();
    priority_to(runner, true);
    if let WaitingFor::DeclareBlockers {
        block_capacities, ..
    } = &runner.state().waiting_for
    {
        for source in sources {
            assert!(
                block_capacities[source].is_none_or(|capacity| capacity >= attackers.len() as u32)
            );
        }
    }
    let assignments: Vec<_> = sources
        .iter()
        .flat_map(|source| attackers.iter().map(move |attacker| (*source, *attacker)))
        .collect();
    runner
        .act(GameAction::DeclareBlockers {
            assignments: assignments.clone(),
        })
        .unwrap();
    let combat = runner.state().combat.as_ref().unwrap();
    for attacker in attackers {
        let actual = combat
            .blocker_assignments
            .get(attacker)
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        assert_eq!(actual.len(), sources.len());
        for source in sources {
            assert!(actual.contains(source));
        }
    }
    for source in sources {
        let actual = &combat.blocker_to_attacker[source];
        assert_eq!(actual.len(), attackers.len());
        for attacker in attackers {
            assert!(actual.contains(attacker));
        }
    }
    for _ in 0..16 {
        match &runner.state().waiting_for {
            WaitingFor::OrderTriggers { triggers, .. } => {
                let order = (0..triggers.len()).collect();
                runner.act(GameAction::OrderTriggers { order }).unwrap();
            }
            WaitingFor::TriggerTargetSelection { .. } => {
                runner
                    .act(GameAction::ChooseTarget {
                        target: Some(engine::types::ability::TargetRef::Player(P0)),
                    })
                    .unwrap();
            }
            WaitingFor::Priority { .. } => return,
            other => panic!("unexpected trigger prompt {other:?}"),
        }
    }
    panic!("trigger placement did not settle");
}

fn setup(
    oracle: &str,
    name: &str,
    count: usize,
    flying: bool,
    mode: TriggerMode,
) -> (GameRunner, Vec<ObjectId>, Vec<ObjectId>) {
    let mut scenario = GameScenario::new_n_player(2, 42);
    scenario.at_phase(Phase::PreCombatMain);
    let first = {
        let mut builder = scenario.add_creature(P0, "First attacker", 1, 1);
        if flying {
            builder.with_keyword(Keyword::Flying);
        }
        builder.id()
    };

    let second = scenario.add_creature(P0, "Second attacker", 1, 1).id();
    let sources: Vec<_> = (0..count)
        .map(|_| {
            scenario
                .add_creature(P1, name, 2, 3)
                .from_oracle_text_with_keywords(
                    if name == "High-Rise Sawjack" {
                        &["Reach"]
                    } else {
                        &[]
                    },
                    oracle,
                )
                .id()
        })
        .collect();
    scenario.add_enchantment_from_oracle(P1, "High Ground", HIGH_GROUND);
    let runner = scenario.build();
    for source in &sources {
        let object = &runner.state().objects[source];
        assert_eq!(object.trigger_definitions.len(), 1);
        assert_eq!(object.trigger_definitions[0].definition.mode, mode);
        assert!(!serde_json::to_string(object)
            .unwrap()
            .contains("Unimplemented"));
    }
    (runner, vec![first, second], sources)
}

fn count(runner: &GameRunner, source: ObjectId) -> usize {
    let count = runner
        .state()
        .stack
        .iter()
        .filter(|entry| entry.source_id == source)
        .count();
    assert!(count > 0, "reach guard: source has actual stack triggers");
    count
}

#[test]
fn bare_blocks_once_for_two_assignments() {
    let (mut runner, attackers, sources) = setup(
        SUSTAINER,
        "Sustainer of the Realm",
        1,
        false,
        TriggerMode::Blocks,
    );
    declare(&mut runner, &attackers, &sources);
    let observed = count(&runner, sources[0]);
    runner.advance_until_stack_empty();
    assert!(runner.state().stack.is_empty());
    let object = &runner.state().objects[&sources[0]];
    eprintln!(
        "MEASURED bare Sustainer triggers={observed}, resolved P/T={:?}/{:?}",
        object.power, object.toughness
    );
    // CR 509.3a: bare blocks triggers once even when blocking multiple creatures.
    assert_eq!(
        (observed, object.power, object.toughness),
        (1, Some(2), Some(5))
    );
}

#[test]
fn qualified_blocks_once_per_attacker() {
    let (mut runner, attackers, sources) =
        setup(WALL, "Wall of Frost", 1, false, TriggerMode::Blocks);
    declare(&mut runner, &attackers, &sources);
    let observed = count(&runner, sources[0]);
    let mut bound_attackers: Vec<_> = runner
        .state()
        .stack
        .iter()
        .filter(|entry| entry.source_id == sources[0])
        .map(|entry| match &entry.kind {
            StackEntryKind::TriggeredAbility {
                trigger_event: Some(GameEvent::BlockersDeclared { assignments }),
                ..
            } => {
                assert_eq!(assignments.len(), 1);
                assert_eq!(assignments[0].0, sources[0]);
                assignments[0].1
            }
            other => panic!("missing qualified attacker binding: {other:?}"),
        })
        .collect();
    bound_attackers.sort();
    let mut expected = attackers.clone();
    expected.sort();
    assert_eq!(bound_attackers, expected);
    runner.advance_until_stack_empty();
    assert!(runner.state().stack.is_empty());
    // CR 509.3b: blocks a creature triggers once for each blocked attacker.
    assert_eq!(observed, 2);
}

#[test]
fn qualified_blocks_only_matching_attacker() {
    let (mut runner, attackers, sources) =
        setup(SAWJACK, "High-Rise Sawjack", 1, true, TriggerMode::Blocks);
    declare(&mut runner, &attackers, &sources);
    let observed = count(&runner, sources[0]);
    runner.advance_until_stack_empty();
    assert!(runner.state().stack.is_empty());
    eprintln!("MEASURED High-Rise Sawjack triggers={observed}");
    // CR 509.3b: only the flying attacker matches this qualified trigger.
    assert_eq!(observed, 1);
    assert_eq!(runner.state().objects[&sources[0]].power, Some(4));
}

#[test]
fn bare_compound_blocks_once_with_player_effect_target() {
    let (mut runner, attackers, sources) = setup(
        CADETS,
        "Goblin Cadets",
        1,
        false,
        TriggerMode::BlocksOrBecomesBlocked,
    );
    assert_eq!(
        runner.state().objects[&sources[0]].trigger_definitions[0]
            .definition
            .valid_target,
        Some(engine::types::ability::TargetFilter::Player)
    );
    declare(&mut runner, &attackers, &sources);
    let observed = count(&runner, sources[0]);
    eprintln!("MEASURED Goblin Cadets triggers={observed}");
    runner.advance_until_stack_empty();
    assert!(runner.state().stack.is_empty());
    // CR 509.3a: the blocker side of bare blocks or becomes blocked triggers once.
    assert_eq!(observed, 1);
}

#[test]
fn bare_blocks_once_for_each_source() {
    let (mut runner, attackers, sources) = setup(
        SUSTAINER,
        "Sustainer of the Realm",
        2,
        false,
        TriggerMode::Blocks,
    );
    declare(&mut runner, &attackers, &sources);
    let observed: Vec<_> = sources
        .iter()
        .map(|source| count(&runner, *source))
        .collect();
    eprintln!("MEASURED two Sustainers source trigger counts={observed:?}");
    runner.advance_until_stack_empty();
    assert!(runner.state().stack.is_empty());
    // CR 509.3a: each source independently triggers once for its declaration.
    assert_eq!(observed, vec![1, 1]);
}

#[test]
fn bare_becomes_blocked_preserves_single_trigger_for_two_blockers() {
    const KARN: &str = "Whenever Karn blocks or becomes blocked, it gets -4/+4 until end of turn.\n{1}: Target noncreature artifact becomes an artifact creature with power and toughness each equal to its mana value until end of turn.";
    let mut scenario = GameScenario::new_n_player(2, 42);
    scenario.at_phase(Phase::PreCombatMain);
    let source = scenario
        .add_creature_from_oracle(P0, "Karn, Silver Golem", 4, 4, KARN)
        .id();
    let blockers: Vec<_> = (0..2)
        .map(|_| scenario.add_creature(P1, "Blocker", 1, 1).id())
        .collect();
    let mut runner = scenario.build();
    let object = &runner.state().objects[&source];
    assert_eq!(
        object.trigger_definitions[0].definition.mode,
        TriggerMode::BlocksOrBecomesBlocked
    );
    assert!(!serde_json::to_string(object)
        .unwrap()
        .contains("Unimplemented"));
    priority_to(&mut runner, false);
    runner
        .act(GameAction::DeclareAttackers {
            attacks: vec![(source, AttackTarget::Player(P1))],
            bands: vec![],
        })
        .unwrap();
    priority_to(&mut runner, true);
    runner
        .act(GameAction::DeclareBlockers {
            assignments: blockers.iter().map(|id| (*id, source)).collect(),
        })
        .unwrap();
    assert_eq!(
        runner.state().combat.as_ref().unwrap().blocker_assignments[&source],
        blockers
    );
    let observed = count(&runner, source);
    runner.advance_until_stack_empty();
    assert!(runner.state().stack.is_empty());
    let object = &runner.state().objects[&source];
    eprintln!(
        "MEASURED Karn attacker-side triggers={observed}, resolved P/T={:?}/{:?}",
        object.power, object.toughness
    );
    // CR 509.3c: bare becomes blocked triggers once despite multiple blockers.
    assert_eq!(
        (observed, object.power, object.toughness),
        (1, Some(0), Some(8))
    );
}

#[test]
fn bebop_attack_sibling_preserves_single_printed_trigger() {
    const BEBOP: &str = "Whenever Bebop & Rocksteady attack or block, sacrifice a permanent unless you discard a card.";
    let mut scenario = GameScenario::new_n_player(2, 42);
    scenario.at_phase(Phase::PreCombatMain);
    let source = scenario
        .add_creature_from_oracle(P0, "Bebop & Rocksteady", 6, 6, BEBOP)
        .id();
    let mut runner = scenario.build();
    let object = &runner.state().objects[&source];
    assert!(object
        .trigger_definitions
        .as_slice()
        .iter()
        .any(|entry| entry.definition.mode == TriggerMode::Attacks));
    assert!(object
        .trigger_definitions
        .as_slice()
        .iter()
        .any(|entry| matches!(entry.definition.mode, TriggerMode::Unknown(_))));
    priority_to(&mut runner, false);
    runner
        .act(GameAction::DeclareAttackers {
            attacks: vec![(source, AttackTarget::Player(P1))],
            bands: vec![],
        })
        .unwrap();
    let observed = count(&runner, source);
    eprintln!(
        "MEASURED Bebop & Rocksteady attack sibling triggers={observed}; block remains Unknown"
    );
    // CR 603.2c: the single declared attacking source triggers its attack ability once.
    assert_eq!(observed, 1);
}

#[test]
fn qualified_blocks_no_matching_attacker() {
    let (mut runner, attackers, sources) =
        setup(SAWJACK, "High-Rise Sawjack", 1, false, TriggerMode::Blocks);
    declare(&mut runner, &attackers, &sources);
    assert!(runner
        .state()
        .stack
        .iter()
        .all(|entry| entry.source_id != sources[0]));
    assert_eq!(runner.state().objects[&sources[0]].power, Some(2));
    // Paired positive fixture proves the qualifier, not the source parse, rejects.
    qualified_blocks_only_matching_attacker();
}

#[test]
fn bare_nonparticipating_source_does_not_fire() {
    let (mut runner, attackers, sources) = setup(
        SUSTAINER,
        "Sustainer of the Realm",
        2,
        false,
        TriggerMode::Blocks,
    );
    declare(&mut runner, &attackers, &sources[..1]);
    assert_eq!(count(&runner, sources[0]), 1);
    assert!(runner
        .state()
        .stack
        .iter()
        .all(|entry| entry.source_id != sources[1]));
}

#[test]
fn bare_empty_assignments_do_not_fire() {
    let (mut runner, attackers, sources) = setup(
        SUSTAINER,
        "Sustainer of the Realm",
        1,
        false,
        TriggerMode::Blocks,
    );
    declare(&mut runner, &attackers, &[]);
    assert!(runner
        .state()
        .stack
        .iter()
        .all(|entry| entry.source_id != sources[0]));
    bare_blocks_once_for_two_assignments();
}

#[test]
fn bare_separate_trigger_definitions_each_fire_once() {
    let (mut runner, attackers, sources) = setup(
        SUSTAINER,
        "Sustainer of the Realm",
        1,
        false,
        TriggerMode::Blocks,
    );
    let object = runner.state_mut().objects.get_mut(&sources[0]).unwrap();
    let definition = object.trigger_definitions[0].definition.clone();
    object.push_printed_trigger(definition);
    declare(&mut runner, &attackers, &sources);
    assert_eq!(count(&runner, sources[0]), 2);
    runner.advance_until_stack_empty();
    assert_eq!(runner.state().objects[&sources[0]].toughness, Some(7));
}

#[test]
fn bare_player_target_nonparticipating_source_does_not_fire() {
    let (mut runner, attackers, sources) = setup(
        CADETS,
        "Goblin Cadets",
        2,
        false,
        TriggerMode::BlocksOrBecomesBlocked,
    );
    declare(&mut runner, &attackers, &sources[..1]);
    // CR 509.3a: only the creature actually declared as a blocker triggers.
    assert_eq!(count(&runner, sources[0]), 1);
    assert!(runner
        .state()
        .stack
        .iter()
        .all(|entry| entry.source_id != sources[1]));
}
