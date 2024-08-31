import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

np.random.seed(42)
n_A = n_B = 5000
conversion_rate_A = 0.12
conversion_rate_B = 0.15

conversions_A = np.random.binomial(1, conversion_rate_A, n_A)
conversions_B = np.random.binomial(1, conversion_rate_B, n_B)

df = pd.DataFrame({
    'Group': ['A'] * n_A + ['B'] * n_B,
    'Conversion': np.concatenate([conversions_A, conversions_B])
})

conversion_summary = df.groupby('Group')['Conversion'].agg([np.mean, np.size, np.sum])
conversion_summary.columns = ['Conversion Rate', 'Sample Size', 'Conversions']
conversion_summary['Conversion Rate'] = conversion_summary['Conversions'] / conversion_summary['Sample Size']
conversion_summary

conversions_A_sum = conversions_A.sum()
conversions_B_sum = conversions_B.sum()
conversion_rate_A = conversions_A_sum / n_A
conversion_rate_B = conversions_B_sum / n_B
pooled_conversion_rate = (conversions_A_sum + conversions_B_sum) / (n_A + n_B)
se_pooled = np.sqrt(pooled_conversion_rate * (1 - pooled_conversion_rate) * (1/n_A + 1/n_B))

z_score = (conversion_rate_B - conversion_rate_A) / se_pooled
p_value = stats.norm.sf(abs(z_score)) * 2

print(f'Z-score: {z_score:.3f}')
print(f'P-value: {p_value:.5f}')

if p_value < 0.05:
    print("The difference in conversion rates between Group A and Group B is statistically significant.")
    print("We reject the null hypothesis.")
else:
    print("The difference in conversion rates between Group A and Group B is not statistically significant.")
    print("We fail to reject the null hypothesis.")

conversion_summary.plot(kind='bar', y='Conversion Rate', legend=False)
plt.title('Conversion Rate Comparison Between Group A and Group B')
plt.ylabel('Conversion Rate')
plt.xlabel('Group')
plt.show()