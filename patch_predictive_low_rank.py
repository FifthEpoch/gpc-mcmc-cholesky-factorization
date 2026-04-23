from pathlib import Path

path = Path('experiments/predictative_low_rank.py')
text = path.read_text()
old = '''    print(f"HMC acceptance rate: {hmc_stats[\"accept_rate\"]:.3f}")

    nu_samples = hmc_stats[\"nu_samples\"]               # (n_samples, train_rank)
    f_train_samples = F @ nu_samples.T                 # (n_train, n_samples)
'''
new = '''    print(f"HMC acceptance rate: {hmc_stats[\"accept_rate\"]:.3f}")

    tau_nu = compute_tau_emcee(hmc_stats[\"nu_samples\"])\n    tau_logp = compute_tau_emcee(hmc_stats[\"logp_trace\"])\n    print(f"HMC tau (nu mean): {tau_nu:.2f}")\n    print(f"HMC tau (logp): {tau_logp:.2f}")\n\n    nu_samples = hmc_stats[\"nu_samples\"]               # (n_samples, train_rank)\n    f_train_samples = F @ nu_samples.T                 # (n_train, n_samples)\n'''
if old not in text:
    raise SystemExit('PRINT BLOCK NOT FOUND')
text = text.replace(old, new, 1)
old_save = '''    np.savez(
        results_path,
        X_test=X_test,
        predictive_prob=predictive_prob,
        predictive_std=predictive_std,
        predictive_latent_mean=predictive_latent_mean,
        predictive_latent_std=predictive_latent_std,
        X_train=X_train,
        y_train=y_train,
        accept_rate=hmc_stats[\"accept_rate\"],
        train_rank=train_rank,
        test_rank=min(120, n_test),
        nugget=nugget,
    )
'''
new_save = '''    np.savez(
        results_path,
        X_test=X_test,
        predictive_prob=predictive_prob,
        predictive_std=predictive_std,
        predictive_latent_mean=predictive_latent_mean,
        predictive_latent_std=predictive_latent_std,
        X_train=X_train,
        y_train=y_train,
        accept_rate=hmc_stats[\"accept_rate\"],
        tau_nu=tau_nu,
        tau_logp=tau_logp,
        train_rank=train_rank,
        test_rank=min(120, n_test),
        nugget=nugget,
    )
'''
if old_save not in text:
    raise SystemExit('SAVE BLOCK NOT FOUND')
text = text.replace(old_save, new_save, 1)
path.write_text(text)
print('patched main and save blocks')
