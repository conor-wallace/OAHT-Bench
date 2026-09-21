import jax

from oaht_bench.models.mlp_actor_critic_agent import MLPActorCriticPolicy, ActorWithDoubleCriticPolicy, \
    ActorWithConditionalCriticPolicy, PseudoActorWithDoubleCriticPolicy, \
    PseudoActorWithConditionalCriticPolicy
from oaht_bench.models.rnn_actor_critic_agent import RNNActorCriticPolicy, RNNActorWithConditionalCriticPolicy, \
    PseudoRNNActorWithConditionalCriticPolicy
from oaht_bench.models.cnn_rnn_actor_critic_agent import CNNRNNActorCriticPolicy, \
    CNNRNNActorWithConditionalCriticPolicy, PseudoCNNRNNActorWithConditionalCriticPolicy
from oaht_bench.models.s5_actor_critic_agent import S5ActorCriticPolicy


def _unwrap_obs_shape(env):
    """The unflattened (W, H, C) grid shape the CNN encoder needs.

    Overcooked-v2's wrapper flattens the observation and exposes only the flat
    Box, but the underlying env keeps ``obs_shape``. Walk the ``.env`` chain
    (LogWrapper -> Overcooked-v2 wrapper -> base env) to the first ``obs_shape``.
    """
    e = env
    while not hasattr(e, "obs_shape") and hasattr(e, "env"):
        e = e.env
    if not hasattr(e, "obs_shape"):
        raise ValueError(
            "A cnn_rnn actor_type requires an env exposing obs_shape (the "
            "unflattened grid); none was found unwrapping the env chain. CNN "
            "actors are only defined for Overcooked-v2."
        )
    return tuple(e.obs_shape)

def initialize_s5_agent(config, env, rng):
    """Initialize an S5 agent with the given config.

    Args:
        config: dict, config for the agent
        env: gymnasium environment
        rng: jax.random.PRNGKey, random key for initialization

    Returns:
        policy: S5ActorCriticPolicy, the policy object
        params: dict, initial parameters for the agent
    """
    # Create the S5 policy with direct parameters
    policy = S5ActorCriticPolicy(
        action_dim=env.action_space(env.agents[0]).n,
        obs_dim=config.get("POLICY_INPUT_DIM", env.observation_space(env.agents[0]).shape[0]),
        d_model=config.get("S5_D_MODEL", 128),
        ssm_size=config.get("S5_SSM_SIZE", 128),
        # d_model=config.get("S5_D_MODEL", 16),
        # ssm_size=config.get("S5_SSM_SIZE", 16),
        ssm_n_layers=config.get("S5_N_LAYERS", 2),
        blocks=config.get("S5_BLOCKS", 1),
        fc_hidden_dim=config.get("S5_ACTOR_CRITIC_HIDDEN_DIM", 1024),
        fc_n_layers=config.get("FC_N_LAYERS", 3),
        # fc_hidden_dim=config.get("S5_ACTOR_CRITIC_HIDDEN_DIM", 64),
        # fc_n_layers=config.get("FC_N_LAYERS", 2),
        s5_activation=config.get("S5_ACTIVATION", "full_glu"),
        s5_do_norm=config.get("S5_DO_NORM", True),
        s5_prenorm=config.get("S5_PRENORM", True),
        s5_do_gtrxl_norm=config.get("S5_DO_GTRXL_NORM", True),
    )

    rng, init_rng = jax.random.split(rng)
    init_params = policy.init_params(init_rng)

    return policy, init_params

def initialize_rnn_agent(config, env, rng):
    """Initialize an RNN agent with the given config.

    Args:
        config: dict, config for the agent
        env: gymnasium environment
        rng: jax.random.PRNGKey, random key for initialization

    Returns:
        policy: RNNActorCriticPolicy, the policy object
        params: dict, initial parameters for the agent
    """
    # Create the RNN policy
    policy = RNNActorCriticPolicy(
        action_dim=env.action_space(env.agents[0]).n,
        obs_dim=config.get("POLICY_INPUT_DIM", env.observation_space(env.agents[0]).shape[0]),
        activation=config.get("ACTIVATION", "tanh"),
        fc_hidden_dim=config.get("FC_HIDDEN_DIM", 64),
        gru_hidden_dim=config.get("GRU_HIDDEN_DIM", 64),
    )

    rng, init_rng = jax.random.split(rng)
    init_params = policy.init_params(init_rng)

    return policy, init_params

def initialize_cnn_rnn_agent(config, env, rng):
    """Initialize a CNN+GRU actor-critic (Overcooked-v2; App. C.1.1).

    The RNN analogue of ``initialize_rnn_agent`` with a convolutional stem; the
    grid shape is read from the env (the flattened obs dim can't be un-flattened
    without it).
    """
    policy = CNNRNNActorCriticPolicy(
        action_dim=env.action_space(env.agents[0]).n,
        obs_dim=config.get("POLICY_INPUT_DIM", env.observation_space(env.agents[0]).shape[0]),
        obs_shape=_unwrap_obs_shape(env),
        activation=config.get("ACTIVATION", "relu"),
        fc_hidden_dim=config.get("FC_HIDDEN_DIM", 128),
        gru_hidden_dim=config.get("GRU_HIDDEN_DIM", 128),
    )
    rng, init_rng = jax.random.split(rng)
    init_params = policy.init_params(init_rng)

    return policy, init_params

def initialize_mlp_agent(config, env, rng):
    """
    Initialize an MLP agent with the given config.
    """
    policy = MLPActorCriticPolicy(
        action_dim=env.action_space(env.agents[0]).n,
        obs_dim=config.get("POLICY_INPUT_DIM", env.observation_space(env.agents[0]).shape[0]),
        activation=config.get("ACTIVATION", "tanh"),
        fc_hidden_dim=config.get("FC_HIDDEN_DIM", 64),
    )
    rng, init_rng = jax.random.split(rng)
    init_params = policy.init_params(init_rng)

    return policy, init_params

def initialize_actor_with_double_critic(config, env, rng):
    """Initialize an actor with double critic with the given config."""
    policy = ActorWithDoubleCriticPolicy(
        action_dim=env.action_space(env.agents[0]).n,
        obs_dim=config.get("POLICY_INPUT_DIM", env.observation_space(env.agents[0]).shape[0]),
        activation=config.get("ACTIVATION", "tanh"),
        fc_hidden_dim=config.get("FC_HIDDEN_DIM", 64),
    )
    rng, init_rng = jax.random.split(rng)
    init_params = policy.init_params(init_rng)

    return policy, init_params

def initialize_pseudo_actor_with_double_critic(config, env, rng):
    """Initialize a pseudo actor with double critic with the given config."""
    policy = PseudoActorWithDoubleCriticPolicy(
        action_dim=env.action_space(env.agents[0]).n,
        obs_dim=config.get("POLICY_INPUT_DIM", env.observation_space(env.agents[0]).shape[0]),
        activation=config.get("ACTIVATION", "tanh"),
        fc_hidden_dim=config.get("FC_HIDDEN_DIM", 64),
    )
    rng, init_rng = jax.random.split(rng)
    init_params = policy.init_params(init_rng)

    return policy, init_params

def initialize_actor_with_conditional_critic(config, env, rng):
    """Initialize an actor with conditional critic with the given config.

    Dispatches on ACTOR_TYPE the same way initialize_rnn_agent's callers do --
    a single fix-point for its caller (CoMeDi.py), which does not build the
    policy directly. The ``.get()`` default keeps a config that omits ACTOR_TYPE
    on the MLP path.
    """
    actor_type = config.get("ACTOR_TYPE", "actor_with_conditional_critic")
    policy_cls = {
        "rnn_actor_with_conditional_critic": RNNActorWithConditionalCriticPolicy,
        "cnn_rnn_actor_with_conditional_critic": CNNRNNActorWithConditionalCriticPolicy,
    }.get(actor_type, ActorWithConditionalCriticPolicy)
    kwargs = dict(
        action_dim=env.action_space(env.agents[0]).n,
        obs_dim=config.get("POLICY_INPUT_DIM", env.observation_space(env.agents[0]).shape[0]),
        pop_size=config["POP_SIZE"],
        activation=config.get("ACTIVATION", "tanh"),
        fc_hidden_dim=config.get("FC_HIDDEN_DIM", 64),
    )
    if policy_cls is RNNActorWithConditionalCriticPolicy:
        kwargs["gru_hidden_dim"] = config.get("GRU_HIDDEN_DIM", 64)
    elif policy_cls is CNNRNNActorWithConditionalCriticPolicy:
        kwargs["obs_shape"] = _unwrap_obs_shape(env)
        kwargs["gru_hidden_dim"] = config.get("GRU_HIDDEN_DIM", 128)
    policy = policy_cls(**kwargs)
    rng, init_rng = jax.random.split(rng)
    init_params = policy.init_params(init_rng)

    return policy, init_params

def initialize_pseudo_actor_with_conditional_critic(config, env, rng):
    """Initialize a pseudo actor with conditional critic with the given config."""
    policy = PseudoActorWithConditionalCriticPolicy(
        action_dim=env.action_space(env.agents[0]).n,
        obs_dim=config.get("POLICY_INPUT_DIM", env.observation_space(env.agents[0]).shape[0]),
        pop_size=config["POP_SIZE"],
        activation=config.get("ACTIVATION", "tanh"),
        fc_hidden_dim=config.get("FC_HIDDEN_DIM", 64),
    )
    rng, init_rng = jax.random.split(rng)
    init_params = policy.init_params(init_rng)

    return policy, init_params

def initialize_pseudo_rnn_actor_with_conditional_critic(config, env, rng):
    """Initialize the RNN analogue of a pseudo actor with conditional critic."""
    policy = PseudoRNNActorWithConditionalCriticPolicy(
        action_dim=env.action_space(env.agents[0]).n,
        obs_dim=config.get("POLICY_INPUT_DIM", env.observation_space(env.agents[0]).shape[0]),
        pop_size=config["POP_SIZE"],
        activation=config.get("ACTIVATION", "tanh"),
        fc_hidden_dim=config.get("FC_HIDDEN_DIM", 64),
        gru_hidden_dim=config.get("GRU_HIDDEN_DIM", 64),
    )
    rng, init_rng = jax.random.split(rng)
    init_params = policy.init_params(init_rng)

    return policy, init_params

def initialize_pseudo_cnn_rnn_actor_with_conditional_critic(config, env, rng):
    """Initialize the CNN+GRU analogue of a pseudo actor with conditional critic
    (CoMeDi's Overcooked-v2 warmup shape)."""
    policy = PseudoCNNRNNActorWithConditionalCriticPolicy(
        action_dim=env.action_space(env.agents[0]).n,
        obs_dim=config.get("POLICY_INPUT_DIM", env.observation_space(env.agents[0]).shape[0]),
        obs_shape=_unwrap_obs_shape(env),
        pop_size=config["POP_SIZE"],
        activation=config.get("ACTIVATION", "relu"),
        fc_hidden_dim=config.get("FC_HIDDEN_DIM", 128),
        gru_hidden_dim=config.get("GRU_HIDDEN_DIM", 128),
    )
    rng, init_rng = jax.random.split(rng)
    init_params = policy.init_params(init_rng)

    return policy, init_params
