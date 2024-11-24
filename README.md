##  Overview

smt-explainability is a package for explaining models trained from smt. 

##  Project Structure

```sh
└── smt-explainability/
    ├── .github
    │   ├── dependabot.yml
    │   └── workflows
    ├── LICENSE
    ├── README.md
    ├── pyproject.toml
    ├── requirements.txt
    ├── setup.py
    ├── smt_explainability
    │   ├── __init__.py
    │   ├── pdp
    │   ├── problems
    │   ├── shap
    │   ├── sobol
    │   └── version.py
    └── tutorial
        ├── Explainability_tools.ipynb
        ├── pdp_cantilever.ipynb
        ├── pdp_wing_weight.ipynb
        ├── shap_cantilever.ipynb
        ├── shap_wing_weight.ipynb
        └── sobol_wing_weight.ipynb
```

###  Project Index
<details open>
	<summary><b><code>SMT-EXPLAINABILITY/</code></b></summary>
	<details> <!-- __root__ Submodule -->
		<summary><b>__root__</b></summary>
		<blockquote>
			<table>
			<tr>
				<td><b><a href='https://github.com/daffarobani/smt-explainability/blob/master/requirements.txt'>requirements.txt</a></b></td>
				<td>Lists the required dependencies for the project, ensuring compatibility and smooth execution of the codebase.</td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/daffarobani/smt-explainability/blob/master/pyproject.toml'>pyproject.toml</a></b></td>
				<td>Defines build system requirements using setuptools for the project.</td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/daffarobani/smt-explainability/blob/master/setup.py'>setup.py</a></b></td>
				<td>- Set up the Python package for SMT explainability, defining metadata like name, version, author, and dependencies<br>- The setup.py file facilitates the distribution and installation of the SMT-explainability package, ensuring compatibility with required dependencies and Python version.</td>
			</tr>
			</table>
		</blockquote>
	</details>
	<details> <!-- tutorial Submodule -->
		<summary><b>tutorial</b></summary>
		<blockquote>
			<table>
			<tr>
				<td><b><a href='https://github.com/daffarobani/smt-explainability/blob/master/tutorial/shap_wing_weight.ipynb'>shap_wing_weight.ipynb</a></b></td>
				<td>Tutorial for using SHAP functionalities for wing weight problem.</td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/daffarobani/smt-explainability/blob/master/tutorial/pdp_wing_weight.ipynb'>pdp_wing_weight.ipynb</a></b></td>
				<td>Tutorial for using PDP functionalities for wing weight problem.</td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/daffarobani/smt-explainability/blob/master/tutorial/sobol_wing_weight.ipynb'>sobol_wing_weight.ipynb</a></b></td>
				<td>Tutorial for using Sobol indices functionalities for wing weight problem.</td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/daffarobani/smt-explainability/blob/master/tutorial/pdp_cantilever.ipynb'>pdp_cantilever.ipynb</a></b></td>
				<td>Tutorial for using PDP functionalities for mixed cantilever beam problem.</td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/daffarobani/smt-explainability/blob/master/tutorial/shap_cantilever.ipynb'>shap_cantilever.ipynb</a></b></td>
				<td>Tutorial for using SHAP functionalities for mixed cantilever beam problem.</td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/daffarobani/smt-explainability/blob/master/tutorial/Explainability_tools.ipynb'>Explainability_tools.ipynb</a></b></td>
				<td>(TO DO)</td>
			</tr>
			</table>
		</blockquote>
	</details>
	<details> <!-- smt_explainability Submodule -->
		<summary><b>smt_explainability</b></summary>
		<blockquote>
			<table>
			<tr>
				<td><b><a href='https://github.com/daffarobani/smt-explainability/blob/master/smt_explainability/version.py'>version.py</a></b></td>
				<td>Defines the version number.</td>
			</tr>
			</table>
            <details>
				<summary><b>problems</b></summary>
				<blockquote>
					<table>
					<tr>
						<td><b><a href='https://github.com/daffarobani/smt-explainability/blob/master/smt_explainability/problems/mixed_cantilever.py'>mixed_cantilever.py</a></b></td>
						<td>Mixed cantilever beam problem.</td>
					</tr>
					</table>
				</blockquote>
			</details>
			<details>
				<summary><b>shap</b></summary>
				<blockquote>
					<table>
					<tr>
						<td><b><a href='https://github.com/daffarobani/smt-explainability/blob/master/smt_explainability/shap/shap_display.py'>shap_display.py</a></b></td>
						<td>Provides ShapDisplay class to display SHAP (SHapley Additive exPlanations) values.</td>
					</tr>
                    <tr>
						<td><b><a href='https://github.com/daffarobani/smt-explainability/blob/master/smt_explainability/shap/shap_feature_importance_display.py'>shap_feature_importance_display.py</a></b></td>
						<td>Provides ShapFeatureImportanceDisplay class to display feature importance based on SHAP values.</td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/daffarobani/smt-explainability/blob/master/smt_explainability/shap/shap_values.py'>shap_values.py</a></b></td>
						<td>Compute SHAP values for model predictions using either the kernel or exact method based on the provided observations and reference data.</td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/daffarobani/smt-explainability/blob/master/smt_explainability/shap/shap_feature_importance.py'>shap_feature_importance.py</a></b></td>
						<td>Compute feature importance based on SHAP values.</td>
					</tr>
					</table>
				</blockquote>
			</details>
			<details>
				<summary><b>pdp</b></summary>
				<blockquote>
					<table>
                    <tr>
						<td><b><a href='https://github.com/daffarobani/smt-explainability/blob/master/smt_explainability/pdp/partial_dependence_display.py'>partial_dependence_display.py</a></b></td>
						<td>Provides PartialDependenceDisplay class to display PDP (Partial Dependence Plot) and ICE (Individual Conditional Expectation).</td>
					</tr>
                    <tr>
						<td><b><a href='https://github.com/daffarobani/smt-explainability/blob/master/smt_explainability/pdp/pd_feature_importance_display.py'>pd_feature_importance_display.py</a></b></td>
						<td>Provides PDFeatureImportanceDisplay to display feature importance based on partial dependence.</td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/daffarobani/smt-explainability/blob/master/smt_explainability/pdp/pd_interaction_display.py'>pd_interaction_display.py</a></b></td>
						<td>Provides PDFeatureInteractionDisplay to display feature interaction scores based on partial dependence.</td>
					</tr>
                    <tr>
						<td><b><a href='https://github.com/daffarobani/smt-explainability/blob/master/smt_explainability/pdp/partial_dependence.py'>partial_dependence.py</a></b></td>
						<td>Computes partial dependence and individual conditional expectation.</td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/daffarobani/smt-explainability/blob/master/smt_explainability/pdp/pd_feature_importance.py'>pd_feature_importance.py</a></b></td>
						<td>Compute feature importance based on partial dependence. </td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/daffarobani/smt-explainability/blob/master/smt_explainability/pdp/pd_interaction.py'>pd_interaction.py</a></b></td>
						<td>Compute pairwise and overall feature interaction scores based on partial dependence.</td>
					</tr>
					</table>
				</blockquote>
			</details>
			<details>
				<summary><b>sobol</b></summary>
				<blockquote>
					<table>
					<tr>
						<td><b><a href='https://github.com/daffarobani/smt-explainability/blob/master/smt_explainability/sobol/sobol_indices_display.py'>sobol_indices_display.py</a></b></td>
						<td>Provides SobolIndicesDisplay to display Sobol indices.</td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/daffarobani/smt-explainability/blob/master/smt_explainability/sobol/sobol_indices.py'>sobol_indices.py</a></b></td>
						<td>Compute Sobol indices.</td>
					</tr>
					</table>
				</blockquote>
			</details>
		</blockquote>
	</details>
</details>
