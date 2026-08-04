"""Construction of the LIAM and MeLIBA baseline agents.

Moved out of ``oaht_bench.agents`` because these are *methods under evaluation*
(§6, trajectory-view family), not agent infrastructure. Keeping them beside the
actor-critic primitives conflated "a network we build things from" with "a
baseline we are measuring".

These factories build the **online** LIAM/MeLIBA agents as jax-aht shipped them.
The benchmark's offline variants follow TAO Appendix F and plug into the shared
DT backbone (§3.1); they will supersede these, which are retained meanwhile as
the reference for what the modeling heads do.
"""

import jax

from oaht_bench.agents.initialize_agents import (
    initialize_mlp_agent,
    initialize_rnn_agent,
    initialize_s5_agent,
)
from oaht_bench.algorithms.liam_agent import LIAMPolicy, initialize_liam_encoder_decoder
from oaht_bench.algorithms.meliba_agent import MeLIBAPolicy, initialize_meliba_encoder_decoder


def initialize_liam_agent(config, env, rng):
    """Initialize the LIAM ego agent with the given config.

    Args:
        config: dict, config for the agent
        env: gymnasium environment
        rng: jax.random.PRNGKey, random key for initialization

    Returns:
        liam: LIAMPolicy, the policy object
        params: tuple, initial parameters for the {encoder, decoder} and policy
    """
    rng, init_encoder_decoder_rng, init_policy_rng = jax.random.split(rng, 3)

    # Initialize the policy based on the specified type
    if config["EGO_ACTOR_TYPE"] == "s5":
        ego_policy, init_ego_params = initialize_s5_agent(config, env, init_policy_rng)
    elif config["EGO_ACTOR_TYPE"] == "mlp":
        ego_policy, init_ego_params = initialize_mlp_agent(config, env, init_policy_rng)
    elif config["EGO_ACTOR_TYPE"] == "rnn":
        ego_policy, init_ego_params = initialize_rnn_agent(config, env, init_policy_rng)

    # Initialize the encoder and decoder for LIAM
    encoder, decoder, init_encoder_decoder_params = initialize_liam_encoder_decoder(config, env, init_encoder_decoder_rng)

    liam = LIAMPolicy(
        policy=ego_policy,
        encoder=encoder,
        decoder=decoder
    )
    params = {'encoder': init_encoder_decoder_params['encoder'],
              'decoder': init_encoder_decoder_params['decoder'],
              'policy': init_ego_params}
    return liam, params

def initialize_meliba_agent(config, env, rng):
    """Initialize the MeLIBA ego agent with the given config.

    Args:
        config: dict, config for the agent
        env: gymnasium environment
        rng: jax.random.PRNGKey, random key for initialization

    Returns:
        meliba: MeLIBAPolicy, the policy object
        params: tuple, initial parameters for the {encoder, decoder} and policy
    """
    rng, init_encoder_decoder_rng, init_policy_rng = jax.random.split(rng, 3)

    # Initialize the policy based on the specified type
    if config["EGO_ACTOR_TYPE"] == "s5":
        ego_policy, init_ego_params = initialize_s5_agent(config, env, init_policy_rng)
    elif config["EGO_ACTOR_TYPE"] == "mlp":
        ego_policy, init_ego_params = initialize_mlp_agent(config, env, init_policy_rng)
    elif config["EGO_ACTOR_TYPE"] == "rnn":
        ego_policy, init_ego_params = initialize_rnn_agent(config, env, init_policy_rng)

    # Initialize the encoder and decoder for LIAM
    encoder, decoder, init_encoder_decoder_params = initialize_meliba_encoder_decoder(config, env, init_encoder_decoder_rng)

    meliba = MeLIBAPolicy(
        policy=ego_policy,
        encoder=encoder,
        decoder=decoder
    )
    params = {'encoder': init_encoder_decoder_params['encoder'],
              'decoder': init_encoder_decoder_params['decoder'],
              'policy': init_ego_params}
    return meliba, params
