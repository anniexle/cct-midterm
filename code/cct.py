import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

# 1. Load the data
def load_data():
    url = "https://raw.githubusercontent.com/joachimvandekerckhove/cogs107s25/refs/heads/main/1-mpt/data/plant_knowledge.csv"
    df = pd.read_csv(url)
    data = df.drop(columns=["Informant"]).values
    return data

# 2. Run Cultural Consensus Theory model
def run_cct_model(data):
    N, M = data.shape
    with pm.Model() as model:
        # Priors
        D = pm.Uniform("D", lower=0.5, upper=1.0, shape=N)
        Z = pm.Bernoulli("Z", p=0.5, shape=M)

        # Compute probability
        D_broadcast = D[:, None]
        p = Z * D_broadcast + (1 - Z) * (1 - D_broadcast)

        # Likelihood
        X = pm.Bernoulli("X", p=p, observed=data)

        # MCMC sampling
        trace = pm.sample(draws=2000, chains=4, tune=1000, target_accept=0.9, return_inferencedata=True)
    return model, trace

# 3. Analyze posterior results
def analyze_results(trace, data):
    N, M = data.shape

    # Summary and convergence
    summary = az.summary(trace, var_names=["D", "Z"])
    print(summary)

    # Posterior means
    mean_D = trace.posterior["D"].mean(dim=["chain", "draw"]).values
    mean_Z = trace.posterior["Z"].mean(dim=["chain", "draw"]).values
    consensus = np.round(mean_Z)

    # Visualize
    az.plot_posterior(trace, var_names=["D"])
    plt.suptitle("Posterior Distributions of Informant Competence", y=1.02)
    plt.show()

    az.plot_posterior(trace, var_names=["Z"])
    plt.suptitle("Posterior Distributions of Consensus Answers", y=1.02)
    plt.show()

    # Most and least competent informants
    most_comp = np.argmax(mean_D)
    least_comp = np.argmin(mean_D)
    print(f"Most competent informant: {most_comp} (mean D = {mean_D[most_comp]:.3f})")
    print(f"Least competent informant: {least_comp} (mean D = {mean_D[least_comp]:.3f})")

    # Majority vote
    majority_vote = np.round(data.mean(axis=0))

    # Compare consensus to majority vote
    comparison = pd.DataFrame({
        "Question": np.arange(1, M+1),
        "Consensus_Z": consensus.astype(int),
        "Majority_Vote": majority_vote.astype(int)
    })
    print("\nComparison of Consensus vs. Majority Vote:")
    print(comparison)

    return mean_D, mean_Z, consensus, majority_vote

# Main execution
if __name__ == "__main__":
    data = load_data()
    model, trace = run_cct_model(data)
    analyze_results(trace, data)
