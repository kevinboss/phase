//! Counter-count quantities bound to a single named object (CR 122.1).
//!
//! "for each +1/+1 counter on the creature it targets" (the activation's chosen
//! target), "for each kind of counter on it" (the object a granted trigger is
//! on), "for each acquired taste counter on this artifact" (a multi-word counter
//! name on the source), and an enters-with clause whose two "for each"
//! conjuncts each place counters. Every card is staged from its verbatim Oracle
//! text (MTGJSON `AtomicCards.json`) and driven through the real pipeline.

use engine::game::combat::AttackTarget;
use engine::game::game_object::AttachTarget;
use engine::game::scenario::{GameRunner, GameScenario, P0, P1};
use engine::game::triggers::drain_order_triggers_with_identity;
use engine::types::ability::TargetRef;
use engine::types::actions::GameAction;
use engine::types::counter::{parse_counter_type, CounterType};
use engine::types::game_state::WaitingFor;
use engine::types::identifiers::ObjectId;
use engine::types::mana::{ManaColor, ManaCost, ManaCostShard, ManaType, ManaUnit};
use engine::types::phase::Phase;
use engine::types::player::PlayerId;
use engine::types::zones::Zone;

const WARRIORS_BLADES: &str = "When this Equipment enters, it deals 3 damage to any target and you gain 3 life.\nEquipped creature gets +2/+1.\nEquip {3}. This ability costs {1} less to activate for each +1/+1 counter on the creature it targets.";
const BLITZBALL_STADIUM: &str = "When this artifact enters, support X. (Put a +1/+1 counter on each of up to X target creatures.)\nGo for the Goal! — {3}, {T}: Until end of turn, target creature gains \"Whenever this creature deals combat damage to a player, draw a card for each kind of counter on it\" and it can't be blocked this turn.";
const MOTH_HERB_ELIXIR: &str = "{T}: Put an acquired taste counter on this artifact.\n{T}, Sacrifice this artifact: You draw two cards and lose 4 life. Then you gain 1 life for each acquired taste counter on this artifact.";
const ULASHT: &str = "Ulasht enters with a +1/+1 counter on it for each other red creature you control and a +1/+1 counter on it for each other green creature you control.\n{1}, Remove a +1/+1 counter from Ulasht: Choose one —\n• Ulasht deals 1 damage to target creature.\n• Create a 1/1 green Saproling creature token.";

fn colorless(n: usize) -> Vec<ManaUnit> {
    (0..n)
        .map(|_| ManaUnit::new(ManaType::Colorless, ObjectId(0), false, vec![]))
        .collect()
}

fn pool(runner: &GameRunner, player: PlayerId) -> usize {
    runner.state().players[player.0 as usize].mana_pool.total()
}

fn hand_len(runner: &GameRunner, player: PlayerId) -> usize {
    runner.state().players[player.0 as usize].hand.len()
}

fn seed_library(scenario: &mut GameScenario, player: PlayerId) {
    for name in ["Library A", "Library B", "Library C", "Library D"] {
        scenario.add_card_to_library_top(player, name);
    }
}

fn acquired_taste() -> CounterType {
    parse_counter_type("acquired taste")
}

/// Pay the activation from the pool and pass priority until the stack is
/// empty. Panics on any prompt it is not taught to answer.
fn drive_to_empty_stack(runner: &mut GameRunner) {
    for _ in 0..40 {
        match runner.state().waiting_for.clone() {
            WaitingFor::OrderTriggers { .. } => {
                drain_order_triggers_with_identity(runner.state_mut());
            }
            WaitingFor::ManaPayment { .. } => {
                runner.act(GameAction::PassPriority).expect("pay from pool");
            }
            WaitingFor::Priority { .. } if runner.state().stack.is_empty() => return,
            WaitingFor::Priority { .. } => {
                runner.act(GameAction::PassPriority).expect("pass priority");
            }
            other => panic!("unexpected prompt: {other:?}"),
        }
    }
    panic!("the stack never emptied");
}

// ── Warrior's Blades: a counter count on the activation's target ─────────────

struct BladesBoard {
    runner: GameRunner,
    blades: ObjectId,
    equip: usize,
    big: ObjectId,
    plain: ObjectId,
}

fn blades_board() -> BladesBoard {
    let mut scenario = GameScenario::new();
    scenario.at_phase(Phase::PreCombatMain);
    let blades = scenario
        .add_artifact_from_oracle(P0, "Warrior's Blades", "")
        .with_subtypes(vec!["Equipment"])
        .from_oracle_text_with_keywords(&["Equip"], WARRIORS_BLADES)
        .id();
    let big = scenario.add_creature(P0, "Big", 2, 2).id();
    scenario.with_counter(big, CounterType::Plus1Plus1, 2);
    let plain = scenario.add_creature(P0, "Plain", 2, 2).id();
    scenario.with_mana_pool(P0, colorless(1));
    let runner = scenario.build();
    let equip = runner.state().objects[&blades]
        .abilities
        .iter()
        .position(|ability| ability.cost_reduction.is_some())
        .expect("reach guard: the equip ability carries its cost rider");
    BladesBoard {
        runner,
        blades,
        equip,
        big,
        plain,
    }
}

/// WB1. CR 601.2c + CR 601.2f + CR 602.2b: two +1/+1 counters on the chosen
/// creature reduce Equip {3} to {1}, priced when the target is chosen.
#[test]
fn warriors_blades_equip_costs_less_per_counter_on_its_target() {
    let mut board = blades_board();
    board
        .runner
        .act(GameAction::ActivateAbility {
            source_id: board.blades,
            ability_index: board.equip,
        })
        .expect("equip is offered: the countered creature makes it affordable");
    board
        .runner
        .act(GameAction::SelectTargets {
            targets: vec![TargetRef::Object(board.big)],
        })
        .expect("Big is an affordable target");
    drive_to_empty_stack(&mut board.runner);

    assert_eq!(pool(&board.runner, P0), 0, "exactly {{1}} was paid");
    assert_eq!(
        board.runner.state().objects[&board.blades].attached_to,
        Some(AttachTarget::Object(board.big))
    );
}

/// WB2. CR 601.2h: the counterless creature leaves Equip at {3}, which one
/// mana can't pay — the activation is reversed to priority with nothing paid.
#[test]
fn warriors_blades_equip_on_a_counterless_target_is_unaffordable() {
    let mut board = blades_board();
    board
        .runner
        .act(GameAction::ActivateAbility {
            source_id: board.blades,
            ability_index: board.equip,
        })
        .expect("reach guard: equip is offered");
    let result = board
        .runner
        .act(GameAction::SelectTargets {
            targets: vec![TargetRef::Object(board.plain)],
        })
        .expect("a reversal is a result, not a rejection");
    assert!(!result.disposition.is_applied(), "typed reversal");
    assert!(matches!(result.waiting_for, WaitingFor::Priority { player } if player == P0));
    assert_eq!(pool(&board.runner, P0), 1, "nothing was paid");
    assert_eq!(
        board.runner.state().objects[&board.blades].attached_to,
        None
    );
    assert!(board.runner.state().stack.is_empty());
}

// ── Blitzball Stadium: kind of counter on the creature the trigger is on ─────

/// BS1. CR 122.1: the granted trigger draws one card per kind of counter on the
/// creature dealing the damage — +1/+1 and oil are two kinds, whatever their
/// numbers.
#[test]
fn blitzball_stadium_grant_draws_per_kind_of_counter_on_the_creature() {
    let mut scenario = GameScenario::new();
    scenario.at_phase(Phase::PreCombatMain);
    let stadium = scenario
        .add_artifact_from_oracle(P0, "Blitzball Stadium", BLITZBALL_STADIUM)
        .id();
    let runner_creature = scenario.add_creature(P0, "Runner", 2, 2).id();
    scenario.with_counter(runner_creature, CounterType::Plus1Plus1, 3);
    scenario.with_counter(runner_creature, parse_counter_type("oil"), 1);
    scenario.with_mana_pool(P0, colorless(3));
    seed_library(&mut scenario, P0);
    seed_library(&mut scenario, P1);
    let mut runner = scenario.build();

    runner
        .act(GameAction::ActivateAbility {
            source_id: stadium,
            ability_index: 0,
        })
        .expect("Go for the Goal! activates");
    // The runner is the only creature, so the engine may settle the single
    // legal target itself.
    if matches!(
        runner.state().waiting_for,
        WaitingFor::TargetSelection { .. }
    ) {
        runner
            .act(GameAction::SelectTargets {
                targets: vec![TargetRef::Object(runner_creature)],
            })
            .expect("target the runner");
    }
    drive_to_empty_stack(&mut runner);

    runner.pass_both_players();
    runner
        .act(GameAction::DeclareAttackers {
            attacks: vec![(runner_creature, AttackTarget::Player(P1))],
            bands: vec![],
        })
        .expect("the runner attacks");
    let outcome = runner.combat_damage();
    assert_eq!(
        outcome.life_delta(P1),
        -5,
        "reach guard: the 5/5 runner connected"
    );
    outcome.assert_hand_drawn(P0, 2);
}

// ── Moth Herb Elixir: a multi-word counter name ──────────────────────────────

/// MH1. CR 122.1: "acquired taste" is one counter name — the first ability
/// adds one.
#[test]
fn moth_herb_elixir_puts_an_acquired_taste_counter_on_itself() {
    let mut scenario = GameScenario::new();
    scenario.at_phase(Phase::PreCombatMain);
    let elixir = scenario
        .add_artifact_from_oracle(P0, "Moth Herb Elixir", MOTH_HERB_ELIXIR)
        .id();
    scenario.with_counter(elixir, acquired_taste(), 2);
    let mut runner = scenario.build();

    runner
        .act(GameAction::ActivateAbility {
            source_id: elixir,
            ability_index: 0,
        })
        .expect("the counter ability activates");
    drive_to_empty_stack(&mut runner);
    assert_eq!(
        runner.state().objects[&elixir]
            .counters
            .get(&acquired_taste()),
        Some(&3)
    );
}

/// MH2. CR 122.1 + CR 608.2h: the sacrificed Elixir's acquired taste counters
/// (as it last existed) set the life gained: draw two, lose 4, gain 3.
#[test]
fn moth_herb_elixir_gains_life_per_acquired_taste_counter() {
    let mut scenario = GameScenario::new();
    scenario.at_phase(Phase::PreCombatMain);
    let elixir = scenario
        .add_artifact_from_oracle(P0, "Moth Herb Elixir", MOTH_HERB_ELIXIR)
        .id();
    scenario.with_counter(elixir, acquired_taste(), 3);
    seed_library(&mut scenario, P0);
    let mut runner = scenario.build();
    let hand_before = hand_len(&runner, P0);

    runner
        .act(GameAction::ActivateAbility {
            source_id: elixir,
            ability_index: 1,
        })
        .expect("the sacrifice ability activates");
    drive_to_empty_stack(&mut runner);

    assert_eq!(runner.state().objects[&elixir].zone, Zone::Graveyard);
    assert_eq!(hand_len(&runner, P0), hand_before + 2);
    assert_eq!(runner.life(P0), 19, "20 - 4 + 3");
}

// ── Ulasht, the Hate Seed: conjoined enters-with "for each" placements ───────

/// UL1. CR 614.1c + CR 122.1: one counter per other red creature you control
/// AND one per other green creature you control; the red-green creature is
/// counted by both (Ulasht ruling), the opponent's red creature by neither.
#[test]
fn ulasht_enters_with_counters_for_red_and_for_green_creatures() {
    let mut scenario = GameScenario::new();
    scenario.at_phase(Phase::PreCombatMain);
    scenario
        .add_creature(P0, "Red Ally", 2, 2)
        .with_color(vec![ManaColor::Red]);
    scenario
        .add_creature(P0, "Green Ally", 2, 2)
        .with_color(vec![ManaColor::Green]);
    scenario
        .add_creature(P0, "Red-Green Ally", 2, 2)
        .with_color(vec![ManaColor::Red, ManaColor::Green]);
    scenario
        .add_creature(P1, "Red Foe", 2, 2)
        .with_color(vec![ManaColor::Red]);
    let ulasht = scenario
        .add_creature_to_hand_from_oracle(P0, "Ulasht, the Hate Seed", 0, 0, ULASHT)
        .with_subtypes(vec!["Hellion", "Hydra"])
        .as_legendary()
        .with_color(vec![ManaColor::Red, ManaColor::Green])
        .with_mana_cost(ManaCost::Cost {
            generic: 2,
            shards: vec![ManaCostShard::Red, ManaCostShard::Green],
        })
        .id();
    scenario.with_mana_pool(
        P0,
        [ManaType::Red, ManaType::Green]
            .into_iter()
            .map(|ty| ManaUnit::new(ty, ObjectId(0), false, vec![]))
            .chain(colorless(2))
            .collect(),
    );
    let mut runner = scenario.build();
    let outcome = runner.cast(ulasht).resolve();
    outcome.assert_zone(&[ulasht], Zone::Battlefield);
    assert_eq!(
        runner.state().objects[&ulasht]
            .counters
            .get(&CounterType::Plus1Plus1),
        Some(&4)
    );
}
