import random
import numpy as np
import bagel as bg

# 1. Build a random sequence of length 50
seq = ''.join(random.choices(list(bg.constants.aa_dict.keys()), k=50))
residues = [bg.Residue(name=aa, chain_ID='A', index=i, mutable=True)
            for i, aa in enumerate(seq)]
chain = bg.Chain(residues=residues)

# 2. Choose an oracle (here ESMFold locally)
esmfold = bg.oracles.ESMFold(use_modal=True, config={"glycine_linker": "GGGG"})

# 3. Define energy terms
energies = [
    bg.energies.PTMEnergy(oracle=esmfold, weight=1.0),
    bg.energies.OverallPLDDTEnergy(oracle=esmfold, weight=1.0),
    bg.energies.HydrophobicEnergy(oracle=esmfold, weight=5.0)
]

# 4. Package into a State and System
state = bg.State(name="design", chains=[chain], energy_terms=energies)
system = bg.System(states=[state])

# 5. Run a simple Monte Carlo minimizer
minimizer = bg.minimizer.SimulatedAnnealing(
    mutator=bg.mutation.Canonical(),
    initial_temperature=0.5,
    final_temperature=0.01,
    n_steps=1000,
    log_frequency=100,
    experiment_name="demo_run"
)
minimizer.minimize_system(system)