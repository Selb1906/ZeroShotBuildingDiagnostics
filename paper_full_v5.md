# Beyond Annual Energy Use Intensity: A Hierarchical Building Performance Evaluation Framework Using Zero-Shot Prediction Error from a Pre-Trained Load Forecasting Foundation Model

---

**Journal:** *Buildings* (MDPI) | **Manuscript type:** Full Research Article

**Keywords:** Building energy benchmarking; Zero-shot forecasting; Transformer; CVRMSE; EUI; ENERGY STAR; Temporal pattern diagnosis; BuildingsBench; Foundation model; BDG-2

---

## Abstract

Annual Energy Use Intensity (EUI) benchmarking, as embodied by the U.S. ENERGY STAR Portfolio Manager, has become the global standard for building energy performance assessment. Yet EUI captures only the aggregate quantity of annual energy consumption, systematically missing the temporal operational patterns that represent the most actionable and immediate efficiency opportunities. We propose a hierarchical, three-level building evaluation framework that integrates EUI-based efficiency scoring with the zero-shot prediction error—specifically the Coefficient of Variation of Root Mean Square Error (CVRMSE)—from a large-scale pre-trained load forecasting model as a complementary temporal pattern consistency metric. The underlying model, TransformerWithGaussian-L from the NREL BuildingsBench suite, was pre-trained on 900,000 simulated building load profiles derived from the U.S. Commercial Buildings Energy Consumption Survey (CBECS) via DOE EnergyPlus simulation, enabling it to encode statistically representative diurnal, weekly, and seasonal energy-use patterns across a diverse range of building archetypes and climate contexts. This training gives the model's prediction error a unique property: when the model fails to predict a building's consumption accurately, the magnitude of that failure reflects not random noise but the building's systematic deviation from population-representative operational norms—a quantity we term *population-referenced pattern atypicality*.

Applied to 611 real buildings from the Building Data Genome Project 2 (BDG-2) dataset spanning 9,247,992 total timesteps, we demonstrate through extensive empirical analysis that EUI and CVRMSE are near-orthogonal (r = −0.029 for all 611 buildings, p = 0.48; r = −0.082 for 583 CBECS-mapped buildings, p = 0.047, R² = 0.007), confirming they capture fundamentally distinct performance dimensions. Our three-level hierarchical framework—(Level 1) EUI × CVRMSE quadrant classification, (Level 2) CVRMSE decomposition into inherent variability-driven versus genuinely atypical components via cross-building regression (R² = 0.700), and (Level 3) Normalized Mean Bias Error (NMBE) directional analysis—provides progressively refined, actionable building diagnosis. Benchmarking against CBECS 2018 Table C14 median EUI thresholds (applied to 583 CBECS-mappable buildings), we demonstrate that 64.7% of ENERGY STAR-certifiable buildings (EUI Score ≥ 75 proxy, n=85) exhibit significant temporal operational irregularities undetectable by annual EUI benchmarking. Conversely, 42.9% of buildings (n=250, Quadrant C) exhibit consistent temporal patterns despite exceeding national median EUI, indicating their efficiency gap is structural rather than operational—directing intervention toward capital investment rather than operational changes. NMBE directional analysis is shown to be statistically informative only for genuinely atypical buildings (Mann-Whitney U, p < 0.0001), providing operational intervention guidance exclusively where it is interpretively valid. Nine best-practice candidate buildings are identified exhibiting systematic under-consumption relative to population-representative expectations, with three recurring operational patterns identified: early evening shutdown, afternoon peak suppression, and delayed morning warm-up. Crucially, the framework requires only metered hourly energy consumption and geographic coordinates as inputs—no floor area, building type, occupant counts, or operating schedules—dramatically lowering deployment barriers relative to existing certification systems. Our work bridges the long-standing gap between aggregate annual benchmarking and actionable hourly-resolution building diagnosis, offering a practical pathway toward temporal pattern-aware performance certification that is broadly deployable from smart meter infrastructure.

**Highlights:**
- EUI and zero-shot CVRMSE are empirically near-independent (r = −0.029 for 611 buildings; r = −0.082, R² = 0.007 for 583 CBECS-mapped), capturing orthogonal performance dimensions
- 64.7% of ENERGY STAR-certifiable buildings exhibit temporal pattern irregularities undetected by EUI alone
- CVRMSE decomposition separates inherent variability (CV-driven, R² = 0.700) from genuine pattern atypicality
- NMBE direction is statistically informative only for genuinely atypical buildings (Mann-Whitney U, p < 0.0001)
- Framework requires only hourly metered energy + lat/lon—no additional metadata needed
- Nine best-practice operational patterns identified for peer replication

---

## 1. Introduction

### 1.1 The Limitations of Annual Energy Benchmarking

The global imperative to decarbonize the built environment has intensified the need for rigorous, actionable building energy performance diagnostics [1,2]. Buildings account for approximately 30% of global final energy consumption and 26% of energy-related greenhouse gas emissions [3], and improving operational efficiency at scale requires diagnostic tools that can identify not only *how much* energy a building consumes but critically *how* it consumes energy across temporal dimensions.

The U.S. ENERGY STAR Portfolio Manager, developed by the Environmental Protection Agency (EPA), is the most widely adopted building energy benchmarking system globally, scoring buildings on a 1–100 scale based on the ratio of actual to predicted Energy Use Intensity (EUI), where the predicted EUI baseline is estimated via weighted least squares (WLS) regression on the Commercial Buildings Energy Consumption Survey (CBECS) [4]. Buildings achieving a score of 75 or above qualify for ENERGY STAR certification—a designation recognized in policy, real estate valuation, and corporate sustainability reporting. However, this framework compresses a year's worth of dynamic operational behavior into a single annual scalar. The aggregation is structurally incapable of detecting a building that achieves a low annual EUI through extended low-demand periods offset by episodic overconsumption, or conversely, one that appears "inefficient" in aggregate but whose temporal operations are as consistent and well-managed as possible given its building system characteristics.

This limitation has practical consequences. A facility manager examining an ENERGY STAR score receives no guidance about *when* anomalous energy use occurs, *what type* of operational pattern drives inefficiency, or *whether* operational adjustments or capital investments are the more appropriate remedy. The score is a verdict without a diagnosis.

Subsequent methodological advances have partially addressed these gaps but have not resolved them. Arjunan et al. [5] introduced EnergyStar++, which improves EUI prediction accuracy through gradient-boosted trees and provides SHAP-based feature importance for interpretability. EnergyStar++ substantially outperforms the ENERGY STAR WLS regression on unseen buildings, but the fundamental temporal limitation persists: it operates at the annual resolution, providing no information about intra-annual or intra-daily operational patterns. The Institute for Market Transformation's *Benchmark 8760* initiative [6] has explicitly advocated for hourly-resolution benchmarking as a next-generation standard capable of capturing demand flexibility, grid interaction, and occupant comfort impacts—yet without prescribing a concrete metric for quantifying hourly pattern consistency. Piscitelli et al. [7], working with the same Building Data Genome Project 2 (BDG-2) dataset used in this study, proposed a multi-KPI framework employing six or more manually designed time series indicators covering consumption level, load variability, peak timing, daily consistency, and seasonal behavior. While comprehensive, this approach requires substantial domain expertise for KPI selection and calibration, resists standardization across building types, and provides no single consolidated signal for portfolio-level screening.

### 1.2 The Two-Dimensional EUI × CV Framework and Its Limitations

A natural extension of single-metric benchmarking is to combine EUI with the Coefficient of Variation (CV = σ/μ) of load in a two-dimensional framework, classifying buildings into four quadrants based on median thresholds [8]. This represents a significant conceptual advance by introducing a temporal variability dimension alongside the aggregate efficiency dimension. However, CV is fundamentally *self-referential*: it measures variability relative to a building's own statistical moments, providing no information about whether the observed variability is *typical* or *atypical* relative to comparable buildings. A public assembly hall with structurally event-driven consumption (high CV due to irregular event scheduling) is indistinguishable from an office building with anomalous HVAC cycling (high CV due to operational failures) when viewed through the CV lens alone.

A more fundamental limitation emerges from the following observation: CV is high for a building if its load is variable, regardless of whether that variability is *predictable*. A sports stadium has high CV because events are irregular—but in practice, building management systems can predict game-day consumption quite accurately because the pattern, while variable in magnitude, follows a consistent schedule-driven structure. CV would flag this building as "irregular"; a sufficiently trained forecasting model would not.

Similarly, a building with anomalous night-time equipment operation may exhibit only moderate CV (the 24-hour mean absorbs the night-time excess), yet the operational pattern clearly deviates from population norms in a diagnostically important way. CV would flag this building as "stable"; a model trained on population-representative patterns would identify the anomaly through elevated prediction error.

These limitations motivate replacing self-referenced CV with a *population-referenced* pattern consistency metric—one that measures deviation not from a building's own history but from what buildings of its type, size, and climate context typically do. This is precisely what zero-shot prediction error from a population-trained model provides.

### 1.3 Foundation Models as Population-Referenced Pattern Benchmarks

A transformative development in time series modeling is the emergence of large-scale pre-trained foundation models capable of zero-shot inference across diverse temporal domains without task-specific fine-tuning [9,10,11]. In the building energy domain, NREL's BuildingsBench [12] introduced the first large-scale benchmark for zero-shot building load forecasting, comprising 900,000 simulated building load profiles generated using EnergyPlus with CBECS-derived building archetypes and ASHRAE standard schedules, alongside a suite of pre-trained Transformer-based forecasting models. These models demonstrate that representations learned from simulated building population statistics generalize to real, unseen buildings—achieving competitive or superior performance relative to building-specific baselines without any fine-tuning.

The BuildingsBench platform also evaluates fine-tuning pre-trained representations on target buildings with limited data (transfer learning) and has demonstrated that synthetically pre-trained models generalize well to real commercial buildings [12]. However, no prior work has systematically exploited the *diagnostic* information contained in the *magnitude* of zero-shot prediction error as a building performance benchmarking signal.

We recognize a critical and previously unexploited property of zero-shot prediction error from such a population-trained model: **when the model—trained to represent the joint distribution of building energy patterns across a diverse, statistically calibrated population—fails to predict a building's consumption, that failure is proportional to the building's deviation from population-representative operational norms.** A building whose consumption pattern lies well within the learned manifold of typical buildings will have low CVRMSE; a building exhibiting genuinely unusual patterns will have high CVRMSE—not because the model is poorly calibrated, but because the building's patterns are population-atypical.

This reframing transforms CVRMSE from an accuracy metric (how well does the model predict this specific building?) to a *population-referenced pattern atypicality indicator* (how much does this building deviate from what buildings typically do?). Unlike CV, which references only a building's own statistical moments, CVRMSE references the full joint distribution of diurnal shapes, seasonal patterns, magnitude-to-variability relationships, and climate-load interactions encoded across 900,000 training examples. A building cannot "explain away" its high CVRMSE by appealing to high inherent variability—the model has seen many high-variability buildings and can predict their patterns. If it cannot predict this building, the pattern is genuinely unusual.

### 1.4 Research Objectives and Contributions

Against this backdrop, this paper makes the following contributions:

**Contribution 1 — Theoretical framing:** We formally establish zero-shot CVRMSE from a population-trained forecasting model as a *population-referenced pattern consistency metric* that is theoretically and empirically distinct from both CV (self-referenced variability) and EUI (aggregate efficiency). We develop the conceptual justification for interpreting model prediction error as building-level diagnostic information rather than model accuracy information.

**Contribution 2 — Empirical independence demonstration:** We demonstrate across 611 real buildings (BDG-2) that EUI and CVRMSE are near-orthogonal (r = −0.029 for all buildings; r = −0.082, R² = 0.007 for 583 CBECS-mapped buildings), establishing their validity as independent diagnostic dimensions and confirming that neither is a proxy for the other.

**Contribution 3 — Hierarchical diagnostic framework:** We develop and validate a three-level hierarchical evaluation framework—(L1) EUI × CVRMSE quadrant classification, (L2) CVRMSE decomposition into CV-driven and genuinely atypical components, and (L3) NMBE directional analysis—that progressively refines diagnosis from building-level classification to actionable hourly-resolution intervention recommendations.

**Contribution 4 — ENERGY STAR blind spot quantification:** Benchmarking against CBECS 2018 Table C14 median EUI thresholds, we empirically quantify that 64.7% of ENERGY STAR-certifiable buildings (EUI Score ≥ 75, n=85) exhibit temporal operational irregularities undetectable by annual EUI alone. Conversely, 42.9% of buildings (Quadrant C, n=250) exhibit consistent temporal patterns despite exceeding national median EUI, indicating structural rather than operational inefficiency. We characterize four systematic types of evaluation reversal when CVRMSE is added to the assessment.

**Contribution 5 — Statistical validation of NMBE informativeness:** We provide rigorous non-parametric validation demonstrating that NMBE direction is statistically meaningful only for genuinely atypical buildings (Mann-Whitney U, p < 0.0001), preventing its misapplication to CV-driven or normally behaving buildings where it is interpretively misleading.

**Contribution 6 — Best practice discovery and characterization:** We identify nine buildings exhibiting systematic under-consumption relative to population expectations and characterize three recurring operational best-practice patterns, providing peer-replicable insights for portfolio managers.

**Contribution 7 — Minimal data requirement:** We demonstrate that the entire pattern diagnostic dimension of the framework operates on hourly metered energy consumption and geographic coordinates only—requiring no floor area, building type, occupant counts, or operating schedules—dramatically reducing deployment barriers relative to ENERGY STAR and EnergyStar++.

The remainder of this paper is organized as follows: Section 2 reviews related work; Section 3 describes the data and model; Section 4 presents the hierarchical methodology; Section 5 reports empirical results; Section 6 discusses implications, comparisons, and limitations; Section 7 concludes with policy implications and future research directions.

---

## 2. Related Work

### 2.1 Building Energy Benchmarking Systems: From Annual to Temporal

Building energy benchmarking has evolved from simple EUI league tables to regression-based systems that control for building characteristics. The ENERGY STAR Portfolio Manager [4] uses WLS regression on CBECS survey data to predict building-type-specific EUI given floor area, climate zone, operating hours, worker density, and plug load intensity, producing percentile scores adjusted for these confounders. The framework has achieved substantial policy uptake—over 240,000 U.S. commercial properties are benchmarked annually, and ENERGY STAR certification is mandatory for large buildings in many U.S. cities under building performance standards.

However, fundamental critiques of single-indicator benchmarking have accumulated. Bordass [13] argued that single indicators systematically mislead because they reduce multidimensional performance to an oversimplified scalar that obscures the distinct contributions of building systems, occupancy, and operations. Scofield [14] demonstrated empirically that ENERGY STAR certification does not reliably predict metered energy savings in practice, partly because annual EUI masks occupancy variability and operational dynamics. ASHRAE's Building EQ [15] and the EU's Smart Readiness Indicator [16] have moved toward multi-dimensional assessment, but these require extensive manual data collection that constrains deployment at scale.

Attempts to introduce temporal resolution into building performance assessment have followed several streams. The IMT's *Benchmark 8760* initiative [6] explicitly called for 8,760-hour benchmarking as necessary for capturing demand flexibility, grid interaction, and occupant comfort—but without prescribing a specific methodology. Granderson et al. [22] developed statistical methods for 15-minute to hourly baseline modeling under ASHRAE Guideline 14 [23] for measurement and verification (M&V) applications, establishing CVRMSE and NMBE as standard accuracy metrics in that context. Our work repurposes these M&V metrics in a benchmarking context, inverting the interpretation: rather than measuring how accurately a model predicts a building (M&V view), we use the prediction error magnitude to characterize how atypical a building is relative to a population (benchmarking view).

### 2.2 Multi-Dimensional and Pattern-Based Frameworks

Park and Miller [17] proposed load shape benchmarking using archetypal profiles derived from k-means clustering on daily normalized load curves across a large, diverse building dataset, conceptually demonstrating that hourly pattern information adds value beyond annual metrics. However, clustering-based approaches provide no connection to a reference population benchmark—cluster membership reveals "which pattern archetype" but not "how far from typical."

Andrews and Jain [18] introduced a two-dimensional framework combining energy efficiency with demand flexibility scores via k-medoids clustering on sub-hourly data from New York City's Local Law 97 disclosure dataset. This framework captures the grid interaction dimension that ENERGY STAR misses, but flexibility is distinct from temporal consistency and requires sub-hourly resolution data not universally available.

Piscitelli et al. [7] represents the most methodologically rigorous multi-KPI temporal approach applied to the BDG-2 dataset used in this study. Their framework manually engineers six key performance indicators: normalized energy consumption, load variability index, peak timing regularity, daily load shape consistency, seasonal pattern regularity, and base load fraction. Applied to BDG-2 buildings, the framework demonstrates the value of multi-dimensional temporal characterization. However, several limitations motivate our alternative approach: (1) the manual KPI design requires domain expertise and introduces designer-specific choices, (2) the six KPIs must be aggregated via ad-hoc weighting schemes, (3) there is no formal connection to a reference building population, and (4) the framework does not distinguish whether irregular patterns are building-type-typical or genuinely anomalous. By using a model trained on 900,000 buildings to produce a single CVRMSE metric, we achieve automatic multi-dimensional temporal integration via the model's learned representations, with a formal connection to a population-calibrated reference distribution.

The EUI × CV quadrant framework described in Section 1.2, despite the aforementioned limitations of CV as a self-referential metric, is the direct conceptual predecessor to ours: we extend it by replacing CV with population-referenced CVRMSE as the second axis, and by introducing the CVRMSE decomposition (Level 2) that resolves the fundamental ambiguity in CV-based classification.

### 2.3 Foundation Models for Building Energy Time Series

Large-scale pre-trained models for time series have advanced rapidly, with models such as Chronos [10], MOMENT [9], and TimesFM [11] demonstrating that representations learned from diverse multi-domain time series corpora support competitive zero-shot inference on unseen tasks and domains. In the building energy domain, NREL's BuildingsBench [12] established the first standardized benchmark for this capability, providing both a 900K-building simulation corpus and a diverse set of real-world evaluation datasets including BDG-2.

The TransformerWithGaussian-L model—our zero-shot forecasting backbone—was shown by Emami et al. [12] to achieve state-of-the-art zero-shot forecasting performance on multiple real building datasets, outperforming naive persistence baselines and competing with or surpassing building-specific ARIMA and LSTM models despite no access to target building data. This generalization capability is precisely what makes the model's zero-shot error informative as a diagnostic signal: the model is calibrated to succeed on typical buildings; its failure identifies atypical ones.

The BuildingsBench evaluation framework includes transfer learning benchmarks where pre-trained models are fine-tuned on target buildings with limited data [12], and broader applications of foundation models in energy systems—including anomaly detection via prediction residuals and occupancy estimation—are an active area of research. However, none of this work uses the zero-shot prediction error magnitude itself as a building performance benchmarking metric—this is the distinctive contribution of the present study.

### 2.4 Positioning Summary

Table 1 positions our framework relative to key prior approaches across four dimensions: temporal resolution, reference population, required inputs, and diagnostic specificity.

**Table 1. Positioning of proposed framework relative to prior work**

| Approach | Resolution | Reference | Required Inputs | Diagnostic Specificity |
|----------|-----------|-----------|----------------|----------------------|
| ENERGY STAR (EPA) [4] | Annual EUI | CBECS regression | Area, hours, workers, climate | How much (aggregate) |
| EnergyStar++ [5] | Annual EUI | CBECS (GBT) | Full CBECS metadata | How much + feature importance |
| Benchmark 8760 [6] | Hourly (proposed) | None specified | 8,760 readings | Advocated but unspecified |
| EUI × CV quadrant [8] | Annual + static CV | Self-referential | Area, load data | How much + how variable |
| Piscitelli et al. [7] | Hourly, multi-KPI | None (relative) | Load data, expert KPIs | Six-dimensional manual |
| Andrews & Jain [18] | Sub-hourly | Peer buildings | Sub-hourly load | Efficiency + flexibility |
| **Ours (this work)** | **Hourly** | **900K-building model** | **Hourly load + lat/lon** | **How much + how patterned + why + which direction** |

### 2.5 Distinction from Fault Detection and Diagnosis (FDD)

Data-driven FDD approaches [25,26] diagnose specific equipment-level faults—stuck valves, sensor drift, refrigerant leaks—using detailed building management system (BMS) subsystem data and labeled fault instances. Our framework operates at a fundamentally different level: whole-building portfolio screening using only aggregate meter data, without requiring system-level instrumentation, fault labels, or building-specific model training. The contribution is not diagnosing *which equipment* is malfunctioning, but identifying *which buildings* exhibit temporal consumption patterns that deviate from population-representative norms—and in which direction. This positions our framework as a portfolio-level screening tool that identifies buildings warranting detailed investigation (including FDD), rather than a replacement for equipment-level fault diagnosis. The two approaches are complementary: our framework can prioritize buildings for FDD deployment, reducing the cost and expertise required for large-scale portfolio management.

---

## 3. Data

### 3.1 Pre-Trained Model: BuildingsBench TransformerWithGaussian-L

#### 3.1.1 Architecture

TransformerWithGaussian-L is a causal Transformer encoder [19] from the BuildingsBench model family [12] that outputs Gaussian predictive distributions over hourly building load. The architecture employs multi-head self-attention with causal masking, enabling it to process variable-length historical context sequences and produce 24-step-ahead probabilistic forecasts. Geographic context (latitude, longitude) is provided as static auxiliary features concatenated to the learned time series embeddings, enabling the model to implicitly condition on climate zone without explicit climate variable inputs.

#### 3.1.2 Training Data: Buildings-900K

The model was pre-trained on Buildings-900K, a corpus of 900,000 one-year (8,760-hour) hourly load profiles generated using EnergyPlus building energy simulation software. The corpus is parameterized to reflect the statistical distribution of the U.S. commercial building stock as characterized by CBECS [21]:

- **Building types:** 11 commercial building archetypes (office, retail, warehouse, education, hotel, healthcare, food service, food sales, strip mall, religious worship, miscellaneous)
- **Climate zones:** 16 U.S. climate zones (ASHRAE 169-2013, spanning 2A through 8A)
- **Vintage:** Representative construction vintages from pre-1980 through 2018, with HVAC system types and envelope properties calibrated accordingly
- **Schedules:** ASHRAE Standard 90.1 reference schedules for each building type, providing typical occupancy, lighting, and equipment load profiles
- **CBECS sampling:** Building geometry, floor area, number of floors, and system type assignments are drawn according to CBECS survey weights to ensure the 900K corpus represents the actual U.S. commercial stock distribution

This training procedure means the model learns not individual building behaviors but the *statistical manifold* of energy patterns consistent with CBECS-characterized U.S. commercial building stock—the joint distribution over diurnal shape, seasonal variation, climate response, and magnitude-to-variability relationships. This is the theoretical foundation for interpreting CVRMSE as population-referenced atypicality.

#### 3.1.3 Zero-Shot Inference Configuration

All predictions in this study are generated in strict zero-shot mode—no fine-tuning on BDG-2 data whatsoever. Model configuration:

- **Input context length:** 168 hours (7 days of historical consumption)
- **Prediction horizon:** 24 hours
- **Sliding window stride:** 24 hours
- **Input features:** (1) energy consumption time series, (2) latitude and longitude as static auxiliary inputs
- **Load normalization:** Box-Cox transformation with λ estimated independently per building from its historical data, applied before model input and inverted for post-prediction metric computation
- **Output:** Gaussian predictive distribution (μ, σ²) at each prediction step; we use the mean prediction μ for CVRMSE and NMBE computation

### 3.2 Evaluation Dataset: Building Data Genome Project 2 (BDG-2)

#### 3.2.1 Dataset Overview

The Building Data Genome Project 2 (BDG-2) [20] is one of the largest publicly available collections of real building energy meter data, released in conjunction with the ASHRAE Great Energy Predictor III (GEPIII) competition. The full dataset comprises 3,053 meters from 1,636 buildings across multiple sites in North America and Europe, covering electricity, chilled water, hot water, and steam meters.

For this study, we applied the following inclusion criteria:

1. **Electricity meters only:** Electricity is the only energy carrier available for all buildings; other carriers have partial coverage and require load conversion assumptions
2. **Valid zero-shot predictions:** Buildings where Box-Cox normalization succeeded (positive mean load) and the model produced no NaN or infinite prediction sequences
3. **Sufficient prediction timesteps:** A minimum of 8,000 valid hourly observation-prediction pairs per building, ensuring reliable CVRMSE and NMBE estimation (equivalent to approximately 333 days)
4. **Floor area availability:** Required for EUI computation (Level 1)

These criteria yield a final study dataset of **611 buildings** with **9,247,992** total observation-prediction timestep pairs, spanning four geographic sites (Bear, Fox, Rat, Panther) across North American climate zones 2C through 6A.

#### 3.2.2 Study Dataset Characteristics

**Table 2. Building type distribution in the study dataset (n = 611)**

| Building Type | Count | Proportion (%) | ENERGY STAR Program Eligible |
|---------------|-------|----------------|------------------------------|
| Education | 254 | 41.6 | Yes (K-12 and university programs) |
| Public Services | 98 | 16.0 | Limited (firefighting, courthouses) |
| Entertainment / Public Assembly | 80 | 13.1 | No |
| Office | 71 | 11.6 | Yes |
| Lodging / Residential | 54 | 8.8 | Yes (hotel/motel) |
| Parking | 15 | 2.5 | No |
| Other (Warehouse, Retail, etc.) | 39 | 6.4 | Partial |
| **Total** | **611** | **100.0** | – |

The dataset is dominated by education buildings (41.6%), reflecting the composition of the BDG-2 sites. All four sites are in North America, providing climate diversity (marine, humid subtropical, humid continental, subarctic) without requiring cross-continental generalization claims.

Metadata coverage varies substantially across fields: floor area (100.0%) and building type (98.7%) are available for nearly all buildings, enabling EUI computation and within-type normalization. Geographic coordinates (latitude/longitude) are available for 85.5% of buildings, providing climate context for the model. Year built is available for 67.2% of buildings but is used only for background analysis. Crucially, number of occupants (14.1%) and operating hours (0.0%) have very low or zero availability. Operating hours—a required input for ENERGY STAR's WLS regression—are entirely unavailable in BDG-2. This limitation motivates our CBECS population-referenced z-score approach for EUI scoring (Section 4.2.1), which requires no building-specific metadata. This demonstrates that our framework's pattern diagnostic dimension (CVRMSE, NMBE) operates with no metadata beyond lat/lon.

### 3.3 Metric Definitions

**Energy Use Intensity (EUI):** Annual electricity consumption divided by gross floor area (sqft), yielding kWh/sqft/year. Computed as (mean hourly load in kWh × 8,760 hours) / sqft. Annual totals are computed by summing available hourly meter readings; for buildings with less than 365 days of data, values are pro-rated assuming uniform seasonal distribution.

#### 3.3.1 Data Consistency Requirement: Electricity-Only Evaluation

This framework evaluates building performance using **Advanced Metering Infrastructure (AMI) electricity meter data**, which is the most widely deployed high-frequency energy data source in commercial building portfolios. To ensure methodologically consistent evaluation, all EUI calculations and reference benchmarks use **electricity consumption only**, excluding natural gas, district heat, fuel oil, and other fuel sources.

**Rationale for electricity-only evaluation:**

1. **AMI data availability:** Smart electricity meters (AMI) provide hourly or sub-hourly resolution at scale. Gas metering is typically monthly or daily, and district energy/fuel oil are rarely sub-metered at all. A framework requiring multi-fuel hourly data would exclude 80%+ of buildings that have only electricity AMI.

2. **Foundation model training data:** The BuildingsBench TransformerWithGaussian-L model was trained on electricity load profiles from the Buildings-900K dataset [12], which contains simulated hourly electricity consumption. The model has learned electricity consumption patterns—not total site energy patterns. Feeding total site EUI to an electricity-trained model introduces category mismatch.

3. **Systematic bias prevention:** Using total site EUI (electricity + gas + other fuels) as a benchmark while evaluating electricity-only meter data creates unfair comparisons:
   - **All-electric buildings** would appear systematically inefficient: their total site EUI equals their electricity EUI (no gas offset), making them look high-EUI compared to mixed-fuel buildings.
   - **Gas-heated buildings** would appear systematically efficient: their electricity EUI is low (lighting/plugs/cooling only), but their total energy consumption (electricity + gas) may be high. These buildings would receive high EUI Scores despite potentially wasteful heating.
   - **Climate bias:** Cold-climate buildings with gas heating would systematically outperform warm-climate all-electric buildings on electricity EUI, even if total energy use is equivalent.

4. **Benchmark consistency:** CBECS 2018 provides separate tables for total site energy and electricity consumption intensity. Table C14 ("Electricity consumption and expenditure intensities") directly publishes quartile distributions of electricity intensity (kWh/sqft) by building type. Since we evaluate electricity meter data, we use Table C14 electricity intensity benchmarks—not total site energy tables.

**Practical implication:** All EUI values in this study refer to **electricity site EUI** in kWh/sqft/year. CBECS 2018 Table C14 provides reference values natively in kWh/sqft, ensuring unit consistency throughout the framework. Similarly, CBECS reference values (Table 3) represent electricity intensity only.

**Generalization to multi-fuel portfolios:** For building portfolios with comprehensive multi-fuel hourly metering, the framework can be extended by: (a) training separate foundation models on gas, steam, or chilled water consumption patterns, or (b) converting all fuel types to total site energy and using CBECS Table E1 total site benchmarks. However, such data is rarely available at portfolio scale, limiting practical applicability.

**Coefficient of Variation (CV):** Standard deviation of hourly load divided by mean hourly load over the observation period:

$$\text{CV}_i = \frac{\sigma_i}{\mu_i}$$

CV is computed from measured consumption data only; no model prediction is involved. CV characterizes a building's inherent load variability relative to its own mean—a self-referential metric.

**Coefficient of Variation of Root Mean Square Error (CVRMSE):** Standardized measure of zero-shot model prediction error:

$$\text{CVRMSE}_i = \frac{\sqrt{\frac{1}{n_i}\sum_{t=1}^{n_i}(y_{i,t} - \hat{y}_{i,t})^2}}{\bar{y}_i}$$

where $y_{i,t}$ is measured consumption at timestep $t$ for building $i$, $\hat{y}_{i,t}$ is the model's zero-shot point prediction, $\bar{y}_i$ is mean measured consumption, and $n_i$ is the number of valid timesteps. CVRMSE normalizes RMSE by the building's mean load, enabling cross-building comparison of pattern deviation regardless of absolute consumption magnitude.

**Normalized Mean Bias Error (NMBE):** Systematic directional bias in model predictions:

$$\text{NMBE}_i = \frac{\frac{1}{n_i}\sum_{t=1}^{n_i}(y_{i,t} - \hat{y}_{i,t})}{\bar{y}_i}$$

Positive NMBE indicates the building systematically consumes *more* than the model predicts (model under-predicts actual; the building is over-consuming relative to population expectations). Negative NMBE indicates the building systematically consumes *less* (the building is under-consuming relative to expectations—potentially indicating efficient operational practices). The NMBE formulation follows ASHRAE Guideline 14 [23] conventions, ensuring consistency with M&V literature.

**Excess CVRMSE (defined in Section 4.3):** The portion of a building's CVRMSE that exceeds what would be predicted from its inherent load variability (CV) alone, capturing the model's unique contribution to pattern atypicality diagnosis.

---

## 4. Methodology: Three-Level Hierarchical Evaluation Framework

### 4.1 Conceptual Architecture

Our framework is designed around three diagnostic questions, each answered at increasing specificity:

- **Level 1:** *Is this building performing well on both the aggregate efficiency dimension (EUI) and the temporal pattern consistency dimension (CVRMSE)?* → Four-quadrant classification
- **Level 2:** *For buildings with irregular temporal patterns: Is this irregularity inherent to the building's use type, or does it represent genuine population-level anomaly?* → CVRMSE decomposition
- **Level 3:** *For genuinely anomalous buildings: Does the anomaly manifest as over-consumption or under-consumption relative to population expectations?* → NMBE directional analysis

The framework is strictly hierarchical—each level's question is only meaningful given the prior level's answer. Applying NMBE analysis to all buildings regardless of CVRMSE classification would produce noisy, uninterpretable signals (a point we validate statistically in Section 5.4).

Figure 1 illustrates the overall framework logic as a decision tree, with building count annotations at each node based on the BDG-2 study dataset.

### 4.2 Level 1: EUI × CVRMSE Quadrant Classification

#### 4.2.1 EUI Score Computation

To enable absolute, population-referenced EUI evaluation, we compute EUI Score using z-score normalization against **CBECS 2018 Table C14 electricity consumption intensity benchmarks** [24]. For building $i$ of type $k$, the EUI Score is computed as:

$$z_i^{(k)} = \frac{\text{EUI}_i - \mu_{\text{CBECS,elec}}^{(k)}}{\sigma_{\text{CBECS,elec}}^{(k)}}$$

$$\text{EUI Score}_i = \Phi(-z_i^{(k)}) \times 100$$

where $\Phi$ is the standard normal cumulative distribution function, $\mu_{\text{CBECS,elec}}^{(k)}$ is the CBECS 2018 Table C14 median electricity EUI for building type $k$ (Table 3, see Section 4.2.1.1), and $\sigma_{\text{CBECS,elec}}^{(k)}$ is the estimated standard deviation. The negative sign in $\Phi(-z)$ ensures that lower EUI (more efficient) yields higher scores.

**Key properties of this scoring method:**

1. **Absolute evaluation:** A building's score is fixed relative to the national CBECS population, not relative to other buildings in the evaluation sample. Adding or removing buildings from BDG-2 does not change any building's score.

2. **National reference:** Score = 50 means the building's electricity EUI equals the CBECS national median for its type—performing at the 50th percentile among all U.S. commercial buildings of that type.

3. **Building type normalization:** Systematic EUI differences across types (e.g., data centers vs. warehouses) are preserved via type-specific reference values, but within-type comparisons are referenced to national norms.

4. **Electricity-only consistency:** Since this framework evaluates AMI electricity meter data exclusively, EUI reference values are drawn from CBECS 2018 Table C14 (electricity consumption intensity only), not total site energy tables (which include gas/other fuels). This ensures fair comparison—all-electric buildings are not penalized, and gas-heated buildings are not artificially advantaged. See Section 3.3.1 for detailed justification.

**Comparison to ENERGY STAR:** Like ENERGY STAR, our method references buildings to a national population baseline (CBECS). Unlike ENERGY STAR's WLS regression, we do not adjust for operating hours, worker density, or plug load within types. This simplification is intentional—our core contribution is the *pattern* dimension (CVRMSE), and we demonstrate in Section 6.3 that the EUI ↔ CVRMSE independence finding is robust to EUI scoring method, including when compared to actual ENERGY STAR scores.

##### 4.2.1.1 CBECS Electricity EUI Reference Values

Table 3 presents the electricity consumption intensity reference thresholds directly from **CBECS 2018 Table C14**. We use CBECS 2018 (rather than 2012) because the evaluation dataset (BDG-2) contains buildings with 2016-2017 meter data, making CBECS 2018 the temporally appropriate reference population.

**CBECS 2018 Table C14** ("Electricity consumption and expenditure intensities") provides comprehensive quartile distributions of electricity intensity by principal building activity, including median, 25th percentile, and 75th percentile values in kWh/sqft. Unlike aggregated consumption tables, C14 directly publishes the intensity distributions we need for benchmarking, eliminating the need for estimation. We use these values directly:

- **Median (kWh/sqft):** Threshold for Score = 50
- **Standard deviation:** Estimated from interquartile range using $\hat{\sigma} = \text{IQR} / 1.35 = (P_{75} - P_{25}) / 1.35$

The IQR-based std estimation is robust and assumes approximately normal distribution within the central 50% of the data—appropriate for population-level EUI distributions which are typically right-skewed but well-behaved in the interquartile range.

**Table 3. Electricity Consumption Intensity Reference (CBECS 2018 Table C14).** All values in kWh/sqft (electricity only; 1 kWh = 3.412 kBtu). Score = 50 corresponds to the C14 median for each type.

| Building Type | Median (kWh/sqft) | P25 | P75 | Est. Std ^1^ |
|---|---|---|---|---|
| Education | 9.6 | 5.8 | 13.1 | 5.4 |
| Office | 10.1 | 6.1 | 15.7 | 7.1 |
| Lodging | 14.5 | 8.2 | 24.5 | 12.1 |
| Public Assembly | 8.2 | 3.7 | 14.9 | 8.3 |
| Public Services | 13.8 | 8.2 | 15.3 | 5.3 |

^1^ Std = (P75 − P25) / 1.35 (IQR-based estimation).

**Temporal alignment rationale:** The foundation model (TransformerWithGaussian-L) was trained on Buildings-900K dataset (simulated using CBECS 2012-era building stock characteristics), but the *evaluation* buildings (BDG-2) have 2016-2017 meter data. For benchmarking, we reference buildings against their contemporaneous population—CBECS 2018 represents the U.S. commercial building stock closest to BDG-2's measurement period. This ensures that a building scoring 50 is compared against national norms from the same era, not norms from 4-5 years earlier.

**Building type coverage:** 583 of 611 BDG-2 buildings map to C14 categories. 28 buildings (Parking, Technology, Utility, Other without specific C14 categories) are excluded from EUI scoring but retain Pattern Score evaluation.

#### 4.2.2 Pattern Score Computation

Within-type z-score normalization of CVRMSE:

$$z_i^{(k)} = \frac{\text{CVRMSE}_i - \mu_{\text{CVRMSE}}^{(k)}}{\sigma_{\text{CVRMSE}}^{(k)}}$$

Pattern Score is mapped from z-score to percentile via the standard normal CDF, with sign inversion so that lower CVRMSE (more consistent pattern) yields higher Pattern Score:

$$\text{Pattern Score}_i = \Phi(-z_i^{(k)}) \times 100$$

The z-score normalization removes between-type CVRMSE differences driven by inherent building type characteristics: parking garages have structurally low CVRMSE (simple, predictable 24/7 or daytime-only patterns), while public assembly buildings have structurally high CVRMSE (event-driven, highly irregular consumption). The Pattern Score reflects within-type relative pattern consistency—directly analogous to ENERGY STAR's type-conditional percentile scoring.

#### 4.2.3 Four-Quadrant Classification

Buildings are classified using Score = 50 as the threshold on both axes, with **inclusive inequality** (≥ 50):

| | **Pattern Score ≥ 50** (temporally consistent) | **Pattern Score < 50** (temporally irregular) |
|---|---|---|
| **EUI Score ≥ 50** (efficient) | **Quadrant A: Excellent** — Efficient and consistent | **Quadrant B: Efficient but Irregular** |
| **EUI Score < 50** (inefficient) | **Quadrant C: Consistent but Inefficient** | **Quadrant D: Needs Improvement** |

**Note on boundary handling:** A score of exactly 50.0 corresponds to the population median and is classified as "at or above median" (favorable). This inclusive threshold follows the standard percentile convention used in ENERGY STAR (≥ 75 for certification).

**Interpretation of the Score = 50 threshold:**

- **EUI Score = 50:** Building's electricity EUI equals the CBECS national median for its type. Buildings with Score ≥ 50 consume no more electricity per square foot than the typical U.S. commercial building of the same type (at or better than median efficiency nationally).

- **Pattern Score = 50:** Building's CVRMSE equals the BDG-2 mean for its type. Since the foundation model was trained on 900K CBECS-representative simulations, Score ≥ 50 indicates the building's temporal pattern is at least as consistent as the typical pattern learned by the model from the training population.

Thus, buildings with both scores ≥ 50 are classified as **Excellent (A)**: efficient relative to the national population *and* operationally consistent relative to population-learned patterns. Buildings with EUI Score ≥ 50 but Pattern Score < 50 are **Efficient but Irregular (B)**: low annual consumption but atypical temporal patterns. Buildings with Pattern Score ≥ 50 but EUI Score < 50 are **Consistent but Inefficient (C)**: predictable operations but high annual consumption. Buildings below 50 on both dimensions are **Needs Improvement (D)**.

**Operational interpretation:**
- **Quadrant A:** Building operations are optimal on both dimensions; maintain current practices
- **Quadrant B:** Annual efficiency is good, but operational patterns deviate from typical; investigate operational controls
- **Quadrant C:** Operational patterns are consistent; efficiency gap likely reflects building system characteristics (aging HVAC, poor envelope, high-density occupancy); prioritize capital investment
- **Quadrant D:** Both dimensions require attention; operational and capital improvements both warranted

### 4.3 Level 2: CVRMSE Decomposition

#### 4.3.1 The CV-CVRMSE Relationship

For buildings in Quadrants B and D (Pattern Score < 50), the elevated CVRMSE may be driven by either: (a) genuinely high inherent load variability (high CV) that naturally challenges prediction regardless of pattern typicality, or (b) a genuinely anomalous pattern that the model cannot capture even accounting for inherent variability.

Distinguishing these cases requires understanding the systematic relationship between CV and CVRMSE across the building population. We fit an ordinary least squares (OLS) regression across all 611 buildings:

$$\text{CVRMSE}_i = \alpha \cdot \text{CV}_i + \beta + \epsilon_i$$

yielding:

$$\widehat{\text{CVRMSE}}_i = 0.541 \times \text{CV}_i - 0.030 \quad (R^2 = 0.700, \; n = 611)$$

Figure 3 visualizes this regression with the 5 pp Excess CVRMSE threshold that separates CV_DRIVEN from ATYPICAL buildings. This relationship reveals that 70% of cross-building variance in CVRMSE is explained by a building's inherent load variability (CV). The coefficient α = 0.541 has a clear physical interpretation: for every unit increase in CV (every percentage point increase in load variability relative to mean), the model's CVRMSE increases by approximately 0.541 percentage points, on average across the population. This is expected—buildings with more variable load are inherently harder to predict, even if their patterns are population-typical.

**Important methodological clarification:** This regression is a *post-hoc cross-building correlation*, not a formula for computing CVRMSE. Each building's CVRMSE is independently computed from its zero-shot prediction errors. The regression describes the population-level relationship between inherent variability and prediction difficulty; the residual $\epsilon_i$ measures each building's deviation from this population relationship—the portion of CVRMSE not explained by inherent variability.

#### 4.3.2 Excess CVRMSE and ATYPICAL Classification

We define:

$$\text{Excess CVRMSE}_i = \text{CVRMSE}_i - (0.541 \times \text{CV}_i - 0.030)$$

Buildings with Excess CVRMSE > 5 percentage points are classified as **ATYPICAL** (genuine pattern deviation beyond what inherent variability would predict); buildings with Excess CVRMSE ≤ 5 pp are classified as **CV_DRIVEN** (elevated CVRMSE primarily explained by inherent load variability).

The 5 pp threshold is selected based on convergent evidence from multiple independent statistical criteria (detailed in Supplementary Material). First, the IQR outlier fence—the universally accepted criterion for identifying statistical outliers (Tukey 1977)—places the upper boundary at Q3 + 1.5 × IQR = 11.6 pp; the adjusted fence for skewed distributions (Hubert & Vandervieren 2008) gives 7.9 pp. Our threshold is positioned conservatively below both fences. Second, Cohen's d for |NMBE| separation between the resulting groups first reaches the "large" effect size threshold (d ≥ 0.8; Cohen 1988) at 4.5 pp, with d = 0.86 at 5 pp. Third, 5 pp is the highest threshold that maintains n ≥ 10 in all three Level 3 directional subcategories (OVER/NEUTRAL/UNDER), ensuring statistically reliable downstream analysis. The intersection of these criteria—large effect size onset (≥ 4.5 pp), maximum feasible threshold for downstream analysis (≤ 5.0 pp), and conservative positioning below the IQR fence—uniquely identifies 5 pp. We acknowledge that this threshold is sample-specific to BDG-2 and requires recalibration on other datasets using the same multi-criteria approach (Section 6.5).

Buildings with Pattern Score ≥ 50 (Quadrants A and C) are classified as **NORMAL**—no CVRMSE decomposition is needed as their pattern consistency is already established.

#### 4.3.3 Interpretation of ATYPICAL vs. CV_DRIVEN

**CV_DRIVEN buildings** have CVRMSE elevated by inherent operational complexity—perhaps they serve variable occupancy types, have multi-use functions, or operate on highly irregular schedules. The model cannot easily predict them because their variability is structurally high, not because their patterns are population-unusual. For these buildings, the high CVRMSE is less immediately actionable: reducing CVRMSE would require reducing inherent variability (e.g., through operational simplification or schedule regularization), which may not align with the building's functional requirements.

**ATYPICAL buildings** exhibit CVRMSE that exceeds what their inherent variability would predict—meaning the model fails to capture their patterns even after accounting for the challenge posed by their variability. These buildings behave in ways the model—trained on 900,000 representative buildings—has not encountered, suggesting operational patterns that deviate from population norms in diagnostically meaningful ways. For these buildings, the high CVRMSE signals a genuine anomaly worthy of operational investigation.

### 4.4 Level 3: NMBE Directional Analysis

For ATYPICAL buildings, the model's systematic prediction error direction (NMBE) provides actionable guidance on the nature of the deviation:

- **OVER-CONSUMING** (NMBE > +2%): The model systematically under-predicts actual consumption. The building consumes more than population-representative expectations for its temporal context. This may indicate after-hours equipment operation, HVAC scheduling failures, plug load proliferation, or other patterns where operational adjustments could reduce consumption toward model expectations.

- **UNDER-CONSUMING** (NMBE < −2%): The model systematically over-predicts. The building consistently consumes less than expected given its temporal context. This potentially indicates operational best practices—aggressive demand management, optimized HVAC scheduling, effective occupancy-linked controls—that other buildings of the same type could emulate.

- **NEUTRAL** (|NMBE| ≤ 2%): Pattern is atypical but without directional bias. The model's errors are symmetric around zero; the atypicality manifests in pattern shape or structure rather than systematic over- or under-consumption. Further investigation of the temporal error structure (hour-of-day NMBE profiles) is warranted.

The ±2% threshold follows ASHRAE Guideline 14 [23] acceptance criteria for M&V models, which require |NMBE| ≤ 5% and CVRMSE ≤ 15% for monthly models. We adopt the more conservative ±2% to identify only buildings with clear, systematic directional bias.

**The restriction to ATYPICAL buildings** is not arbitrary—it is statistically validated. We demonstrate in Section 5.4 that NMBE is systematically non-zero only for ATYPICAL buildings; for NORMAL and CV_DRIVEN buildings, NMBE is near zero and statistically indistinguishable from noise. Applying NMBE interpretation to non-ATYPICAL buildings would generate spurious recommendations.

---

## 5. Results

### 5.1 Baseline Statistics: CVRMSE Distribution and Its Drivers

Table 4 presents the CVRMSE distribution across building types. Parking garages exhibit the lowest median CVRMSE (7.7%), consistent with their simple, predictable daytime-operation-only patterns. Entertainment and public assembly buildings exhibit the highest median CVRMSE (22.3%), reflecting their event-driven, irregular consumption schedules. This between-type variation motivates the within-type z-score normalization for Pattern Score.

**Table 4. CVRMSE distribution by building type (n = 611)**

| Building Type | n | Mean CVRMSE | Median CVRMSE | Std. Dev. | IQR |
|---------------|---|-------------|---------------|-----------|-----|
| Parking | 15 | 10.7% | 7.7% | 8.4% | 5.8% |
| Lodging | 54 | 14.5% | 10.9% | 11.5% | 6.7% |
| Office | 71 | 16.7% | 14.2% | 8.4% | 8.8% |
| Public Services | 98 | 17.3% | 16.0% | 8.3% | 8.3% |
| Education | 254 | 17.0% | 14.8% | 10.8% | 10.4% |
| Entertainment / Assembly | 80 | 28.9% | 22.3% | 32.7% | 16.4% |
| Other | 39 | 20.8% | 16.1% | 22.9% | 14.2% |
| **All buildings** | **611** | **18.4%** | **15.1%** | **16.5%** | **11.1%** |

Entertainment/Assembly exhibits notably high standard deviation (32.7%, exceeding its mean of 28.9%), reflecting a small number of high-CVRMSE outliers with event-driven consumption; the within-type z-score normalization for Pattern Score accounts for such skewness.

The CV-CVRMSE regression (R² = 0.700) confirms that inherent load variability is the primary driver of cross-building CVRMSE variation. Among the 185 buildings with CVRMSE > 20%, three distinct (non-mutually-exclusive) causal mechanisms contribute: high inherent variability (87.6%), small mean load inflating the normalized metric (47.6%), and genuine pattern deviation (33.5%). The detailed causal decomposition is provided in Appendix D.

### 5.2 EUI and CVRMSE Are Empirically Independent

The Pearson correlation between raw EUI and raw CVRMSE across all 611 buildings is r = −0.029 (p = 0.48); on the 583 CBECS-mapped subset, r = −0.082 (p = 0.047, R² = 0.007). Although the latter is marginally significant, EUI explains less than 1% of CVRMSE variance, confirming near-independence. The correlation between EUI Score (C14 median reference) and Pattern Score across 583 CBECS-mapped buildings is r = −0.291 (p < 0.001). The weak negative association at the score level—explaining only 8.5% of variance—reflects a modest tendency for very energy-intensive buildings to exhibit irregular patterns, possibly due to oversized HVAC systems with excessive cycling. However, the explained variance is far insufficient for either metric to substitute for the other; the vast majority of variation in each score is orthogonal to the other dimension.

Figure 2 (EUI Score vs. Pattern Score scatter plot, n = 583) illustrates this near-independence, with a relatively uniform distribution of buildings across the two-dimensional score space rather than a diagonal concentration that would indicate collinearity.

This finding has a critical implication: CVRMSE cannot be inferred from EUI, and EUI cannot be inferred from CVRMSE. An assessment based on either dimension alone will systematically miss the information provided by the other. Two buildings with identical EUI Scores may have Pattern Scores spanning the full 0–100 range; two buildings with identical Pattern Scores may have EUI Scores spanning the full range. Both dimensions must be measured.

### 5.3 Level 1: Quadrant Distribution

**Table 5. Level 1 quadrant classification results (n = 583, CBECS 2018 Table C14 median benchmark)**

| Quadrant | n | % | Median EUI Score | Median Pattern Score | Primary Recommendation |
|----------|---|---|-----------------|---------------------|------------------------|
| A: Excellent | 122 | 20.9 | 66 | 67 | Maintain current operations |
| B: Efficient but Irregular | 107 | 18.4 | 76 | 29 | Investigate operational controls |
| C: Consistent but Inefficient | 250 | 42.9 | 9 | 70 | Capital investment (systems/envelope) |
| D: Needs Improvement | 104 | 17.8 | 19 | 32 | Operational + capital review |
| **Total** | **583** | **100.0** | – | – | – |

**Note:** 583 buildings represent CBECS-mappable types only (Education, Office, Lodging, Retail, Public Assembly, Public Services, Warehouse, Healthcare, Food Service, Worship). 28 buildings excluded (Parking, Technology, Utility, Other without CBECS 2018 C14 category mapping).

**Quadrant C (Consistent but Inefficient) is the most populous category at 42.9%** (n=250), substantially higher than when using within-sample relative ranking. This finding reveals a critical insight: when BDG-2 buildings are benchmarked against the CBECS 2018 Table C14 national median rather than each other, over 42% exhibit higher-than-national-median electricity consumption while maintaining operationally consistent patterns. For these buildings, operational interventions (schedule changes, setpoint adjustments, BMS optimization) are unlikely to produce large efficiency gains; the framework correctly directs attention toward capital improvements in building systems or envelope. This distinction—between operational and capital problems—is precisely what EUI-only benchmarking cannot make.

The absolute CBECS 2018 C14 benchmark reveals that **39.3% of BDG-2 buildings perform at or below the national median electricity EUI** (EUI Score ≥ 50, n=229). This is substantially lower than the 50% expected from a nationally representative sample, suggesting that the BDG-2 dataset—drawn from four U.S. university campuses—may have systematically higher electricity intensity than the broader commercial building stock. Possible explanations include building vintage (many campus buildings predate modern energy codes), operational hours (24/7 research facilities), plug load density (laboratory equipment, servers), and HVAC requirements (ventilation for labs).

Quadrant B (18.4%, n=107) represents the ENERGY STAR blind spot: these buildings appear efficient in annual EUI terms but exhibit irregular temporal patterns. As we demonstrate in Section 5.6, a majority of ENERGY STAR-certifiable buildings (EUI Score ≥ 75) exhibit pattern irregularities undetectable by annual EUI alone.

Building type patterns in quadrant distribution (Appendix E) reveal systematic differences in operational consistency. Lodging facilities exhibit the highest Quadrant A proportion (55.6%), consistent with their operationally regular, schedule-driven patterns. Public Assembly buildings show the highest Quadrant C proportion (62.5%), indicating consistent temporal patterns despite high electricity intensity. Education buildings exhibit the highest Quadrant B proportion among listed types (22.8%), likely reflecting strong academic calendar seasonality and diverse sub-type composition (K-12 vs. university research facilities).

### 5.4 Level 2: CVRMSE Decomposition and Statistical Validation

Table 6 presents the complete Level 2 classification across 583 CBECS-mapped buildings using CBECS 2018 C14 absolute thresholds.

**Table 6. Level 2 classification results (n = 583, CBECS 2018 C14)**

| | NORMAL (Pattern ≥ 50) | CV_DRIVEN (Excess ≤ 5 pp) | ATYPICAL (Excess > 5 pp) | Row Total |
|--|:--------------------:|:------------------------:|:------------------------:|:---------:|
| **A: Excellent** | 122 | 0 | 0 | 122 |
| **B: Efficient but Irregular** | 0 | 81 | 26 | 107 |
| **C: Consistent but Inefficient** | 250 | 0 | 0 | 250 |
| **D: Needs Improvement** | 0 | 72 | 32 | 104 |
| **Column Total** | **372** | **153** | **58** | **583** |

Of the 211 buildings in Quadrants B and D, 153 (72.5%) are CV_DRIVEN: their high CVRMSE is substantially explained by inherent load variability. Only 58 buildings (27.5% of B+D; 9.9% of 583) are ATYPICAL—exhibiting pattern deviations genuinely beyond what their inherent variability would predict.

#### 5.4.1 Statistical Validation of NMBE Informativeness

The decision to apply NMBE directional analysis only to ATYPICAL buildings requires formal statistical justification. We compare NMBE distributions across three groups (Table 7).

**Table 7. NMBE characteristics by CVRMSE group (n = 583)**

| CVRMSE Group | n | Mean Abs. NMBE | % with Abs. NMBE > 5% | Median Abs. NMBE |
|-------------|---|:--------------:|:---------------------:|:----------------:|
| NORMAL (Pattern ≥ 50) | 372 | 0.76% | 0.3% | 0.55% |
| CV_DRIVEN (high CVRMSE, low Excess) | 153 | 1.41% | 1.3% | 1.11% |
| ATYPICAL (high Excess CVRMSE) | 58 | 5.29% | 31.0% | 2.68% |

Mann-Whitney U test comparing |NMBE| distributions between CV_DRIVEN and ATYPICAL groups: **U = 6,326, p < 0.0001** (two-sided). This highly significant result confirms that ATYPICAL buildings exhibit systematically larger and more directionally consistent prediction bias than CV_DRIVEN buildings. For NORMAL and CV_DRIVEN buildings, prediction errors are symmetric around zero—the model struggles to predict magnitudes but does not systematically over- or under-predict direction. For ATYPICAL buildings, prediction errors carry a systematic directional signal that is operationally interpretable.

This statistical validation demonstrates that applying NMBE analysis to non-ATYPICAL buildings would generate misleading guidance. A CV_DRIVEN building with NMBE = +3% should not be interpreted as "over-consuming relative to population expectations"; the model's ±3% error is within the noise range for that building's variability level. Only when the building's CVRMSE exceeds what CV predicts—when there is genuine unexplained pattern deviation—does the NMBE direction carry interpretable signal.

### 5.5 Level 3: NMBE Direction for ATYPICAL Buildings

**Table 8. NMBE directional classification for ATYPICAL buildings (n = 58)**

| Direction | Threshold | Count | % | Mean NMBE | Mean Excess CVRMSE |
|-----------|-----------|-------|---|-----------|-------------------|
| OVER-CONSUMING | NMBE > +2% | 25 | 43.1 | +10.1% | 20.5 pp |
| NEUTRAL | Abs. NMBE ≤ 2% | 23 | 39.7 | −0.2% | 10.2 pp |
| UNDER-CONSUMING | NMBE < −2% | 10 | 17.2 | −3.6% | 17.5 pp |
| **All ATYPICAL** | – | **58** | **100.0** | **+3.7%** | **15.9 pp** |

OVER-CONSUMING buildings exhibit the highest mean Excess CVRMSE (20.5 pp), indicating that the most directionally biased buildings also deviate most strongly from population-expected patterns—the directional bias and pattern atypicality are correlated, reinforcing the diagnostic signal.

Over-consuming ATYPICAL buildings (25 buildings, 43.1% of ATYPICAL) represent the highest-priority operational intervention targets: these buildings deviate from population-typical patterns *and* systematically consume more than the model predicts for their temporal context. The mean NMBE of +10.1% suggests these buildings consume approximately 10% more energy per hour than a comparable population-typical building would in the same temporal context. Reducing consumption toward model expectations—through HVAC scheduling optimization, after-hours load management, or occupancy-linked controls—represents a quantifiable, population-referenced efficiency target.

Under-consuming ATYPICAL buildings (10 buildings, 16.9% of ATYPICAL) are best-practice candidates analyzed in Section 5.7. Their negative NMBE (mean −3.6%) indicates systematic under-consumption relative to expectations—genuine energy saving that the model, calibrated to typical buildings, did not anticipate.

**Worst-case example — Fox_warehouse_Lorretta (Warehouse, 3,877 sqft):**
- Level 1: EUI Score 0, Pattern Score 2 → Quadrant D: Needs Improvement
- Level 2: CV = 1.75, Expected CVRMSE = 91.4%, Actual CVRMSE = 144.4%, Excess = 53.1 pp → ATYPICAL
- Level 3: NMBE = +23.3% → OVER-CONSUMING
- Hourly analysis: Worst over-consumption at 20:00 (23.3 kWh above model expectation on average)
- **Action:** Comprehensive energy audit with focus on evening operations; review equipment operating after 18:00

**Efficient-but-irregular example — Bear_education_Alfredo (Education, 6,564 sqft):**
- Level 1: EUI Score 96, Pattern Score 3 → Quadrant B: Efficient but Irregular
- Level 2: CV = 0.50, Expected CVRMSE = 24.0%, Actual CVRMSE = 39.6%, Excess = 15.8 pp → ATYPICAL
- Level 3: NMBE = +19.4% → OVER-CONSUMING
- **Interpretation:** This building achieves very low annual EUI (top 4% nationally for its type) yet exhibits severely atypical temporal patterns with systematic over-consumption of 19.4% relative to model expectations. ENERGY STAR would certify this building; our framework reveals hidden operational irregularities warranting investigation.
- **Action:** Review operational schedule; investigate after-hours and weekend energy use patterns

### 5.6 ENERGY STAR Blind Spot Analysis

We simulate ENERGY STAR eligibility using within-type EUI percentile score ≥ 75 as a proxy for certification eligibility, benchmarked against CBECS 2018 Table C14 median EUI thresholds. Of the 583 CBECS-mapped buildings:

- **85 buildings** achieve EUI Score ≥ 75 (ENERGY STAR-certifiable by EUI criterion, 14.6%)
- Of these, **55 buildings (64.7%)** have Pattern Score < 50 (irregular temporal operations)
- Of the 55 irregular ENERGY STAR-certifiable buildings, **16 (29.1%)** are ATYPICAL—exhibiting genuine, non-CV-explained pattern anomalies
- The remaining 39 of 55 are CV_DRIVEN—their pattern irregularity is largely inherent to their use type

Conversely, **250 buildings (Quadrant C, 42.9% of CBECS-mapped buildings)** have EUI Score < 50 (exceeding CBECS 2018 C14 median) yet Pattern Score ≥ 50 (consistent temporal operations). These buildings exhibit mean CVRMSE substantially below the full-sample mean, with consistent load patterns that the model can predict accurately. Their higher-than-median annual EUI reflects structural characteristics (aging HVAC equipment, high-density occupancy, poor envelope insulation, high-EUI end uses) rather than operational inconsistency.

The shift from relative within-sample ranking to CBECS 2018 C14 absolute benchmarks produces a notable asymmetry: only 39.3% of BDG-2 buildings achieve at-or-below-median electricity EUI (229/583), versus the 50% expected if BDG-2 were nationally representative. This reflects the university campus characteristics of BDG-2 buildings—research facilities with 24/7 operations, laboratory equipment, and high computing loads—which systematically exceed national commercial building medians for their building types.

**The policy implication is significant:** For the 250 Consistent-but-Inefficient (Quadrant C) buildings—representing 42.9% of the CBECS-mapped portfolio when benchmarked against CBECS 2018 C14 median thresholds—operational changes such as behavior campaigns, schedule adjustments, or BMS reprogramming will yield limited efficiency gains because these buildings are already operating as consistently as possible within their physical constraints. Capital investment (equipment replacement, envelope upgrades, system modernization) is the appropriate intervention, a recommendation that EUI-alone assessment cannot make but that our framework's pattern dimension immediately clarifies.

### 5.7 Four Systematic Reversal Cases

We document four types of assessment reversal that occur when CVRMSE is added to the evaluation framework. Cases 1 and 2 represent reversals relative to a CV-based variability assessment; Cases 3 and 4 are equivalent to Quadrants B and C, respectively (Table 5), representing reversals relative to EUI-only assessment.

**Table 9. Assessment reversals: CV-based vs. population-referenced CVRMSE**

| Case | Prior CV-Based Assessment | n | % of 583 | Our Framework Re-evaluation |
|------|--------------------------|---|----------|----------------------------|
| 1: High CV, good pattern | CV high → "irregular" | 102 | 17.5% | CVRMSE low → Predictable variability; NOT anomalous |
| 2: Low CV, bad pattern | CV low → "stable" | 23 | 3.9% | CVRMSE high → Hidden atypicality detected |

In addition, 107 buildings (Quadrant B, 18.4%) receive favorable EUI assessments but exhibit temporal irregularities detected only by CVRMSE, and 250 buildings (Quadrant C, 42.9%) exceed the national median EUI yet exhibit well-operated temporal patterns, directing intervention toward capital investment rather than operational changes. Combining all four reversal types, 415 buildings (71.2%) receive substantively different assessments when CVRMSE is incorporated—computed as the non-overlapping union of Cases 1–4.

#### Case 1: High CV but Low CVRMSE — Predictable Variability (102 buildings, 17.5%)

These buildings have CV above their type median—a CV-based framework would classify them as "irregular"—but Pattern Score ≥ 50, meaning the model can predict their patterns effectively despite the high variability. Mean characteristics: CV = 0.448 (above-median), CVRMSE = 15.9% (below-sample mean).

The key insight is that *predictable high variability is not anomalous*. A sports stadium has highly variable daily load (game days vs. non-game days), but a model trained on stadiums can predict which days are game days from the preceding week's context signal. CV is high; CVRMSE is low because the pattern is population-typical.

**Example:** Rat_office_Ramiro (Office, 89,315 sqft): CV = 0.363 (above office median), CVRMSE = 8.8%, Pattern Score = 83. This office exhibits strong weekday/weekend differentials (high CV) that follow a highly regular weekly schedule (low CVRMSE). A CV-based approach would flag this as irregular; our model recognizes it as a textbook office schedule pattern.

#### Case 2: Low CV but High CVRMSE — Hidden Atypicality (23 buildings, 3.9%)

These buildings have CV below their type median—a CV-based approach would classify them as "stable"—but Pattern Score < 50, indicating the model cannot accurately predict their patterns despite the apparently low variability. Critically, 16 of 23 (69.6%) are ATYPICAL (Excess CVRMSE > 5 pp).

The mechanism is likely that these buildings have a *flat but anomalous* consumption profile: they consume at an unusually steady rate but at a level and temporal structure that the model, calibrated to typical buildings, does not expect. A building that runs HVAC at near-constant capacity regardless of occupancy or weather will have low CV (nearly constant load) but will confound the model, which expects weather- and occupancy-driven variation.

**Example:** Rat_office_Avis (Office): CV = 0.359 (below median), CVRMSE = 34.5%, Pattern Score = 3, Excess CVRMSE = 18.2 pp (ATYPICAL). Despite apparently low variability, this building is deeply anomalous by population standards—the model's near-constant failure to predict it suggests systematic operational irregularities hidden from CV-based analysis.

#### Case 3: Low EUI but High CVRMSE — Efficient but Temporally Irregular (107 buildings, 18.4%)

These buildings achieve EUI Score ≥ 50 (at or below CBECS 2018 C14 median) but Pattern Score < 50 (Quadrant B). They represent the ENERGY STAR blind spot: EUI-based assessment recognizes their low annual energy intensity but does not flag the temporal irregularities that our model detects. Of these 107 buildings, 24.3% (26 buildings) are ATYPICAL—exhibiting non-CV-explained pattern anomalies that may represent actionable operational inefficiencies.

The policy implication: ENERGY STAR certification provides a necessary but insufficient signal. A certified building may have hidden operational pattern issues that, if addressed, could further reduce energy consumption or improve demand flexibility—benefits invisible to annual EUI.

#### Case 4: High EUI but Low CVRMSE — Above-Median EUI but Well-Operated (250 buildings, 42.9%)

These buildings have EUI Score < 50 (exceeding CBECS 2018 C14 median) but Pattern Score ≥ 50 (consistent patterns, Quadrant C). The model predicts their consumption accurately, indicating their temporal patterns are population-typical.

**Example:** Panther_office_Lois (Office, EUI = 102.3 kWh/sqft/year, CVRMSE = 5.6%, Pattern Score = 91). Despite EUI nearly ten times the CBECS 2018 C14 office median (10.1 kWh/sqft), this building's consumption pattern is highly predictable—the model achieves lower CVRMSE than 91% of office buildings. High EUI reflects characteristics specific to this building (possibly dense occupancy, 24/7 operation, aging HVAC with high parasitic loads, or extensive server infrastructure) rather than operational failures. Behavioral interventions and schedule optimization will yield minimal gains; only capital investment in building systems would materially improve EUI.

This reversal has important practical consequences: energy managers who receive an ENERGY STAR low score for such a building should not invest in behavioral change programs or BMS reprogramming—these will achieve little. The pattern score directs attention to the correct intervention category.

### 5.8 Best Practice Discovery: ATYPICAL Under-Consuming Buildings

Of the 10 UNDER-CONSUMING ATYPICAL buildings (NMBE < −2%, Excess CVRMSE > 5 pp), 9 are confirmed as genuine best-practice candidates after excluding 1 building with mean load < 5 kWh/hr (where CVRMSE inflation from a small denominator creates potentially spurious results):

**Table 10. Best practice candidate buildings (n = 9, confirmed genuine, CBECS 2018 C14)**

| Building | Type (Subtype) | Sqft | NMBE | Peak Hourly Saving | Identified Pattern |
|----------|---------------|------|------|-------------------|-------------------|
| Panther_office_Danica | Office | 2,260 | −5.2% | −0.7 kWh/hr (12:00) | Peak suppression |
| Panther_education_Scarlett | Education (Classroom) | 29,469 | −4.5% | −2.0 kWh/hr (12:00) | Peak suppression |
| Bear_assembly_Beatrice | Public Assembly | 30,160 | −3.4% | −3.9 kWh/hr (08:00) | Delayed morning start |
| Panther_education_Janis | Education (Research) | 10,932 | −3.0% | −2.1 kWh/hr (09:00) | Delayed morning start |
| Panther_office_Brent | Office | 10,743 | −2.9% | −1.1 kWh/hr (05:00) | Overnight minimization |
| Fox_assembly_Boyce | Public Assembly (Fitness) | 167,066 | −2.7% | −12.9 kWh/hr (16:00) | Peak suppression |
| Fox_assembly_Bradley | Public Assembly (Stadium) | 809,530 | −2.5% | −8.6 kWh/hr (07:00) | Delayed morning start |
| Rat_office_Avis | Office | 474,680 | −2.5% | −40.6 kWh/hr (23:00) | Evening shutdown |
| Bear_education_Derek | Education | 68,146 | −2.1% | −6.6 kWh/hr (23:00) | Evening shutdown |

Three distinct operational best-practice patterns emerge from hourly prediction residual analysis (Figure 4):

**Pattern 1 — Peak Suppression (Panther_office_Danica, Panther_education_Scarlett, Fox_assembly_Boyce):** These buildings systematically under-consume during midday and afternoon hours (09:00–16:00) relative to model expectations. Fox_assembly_Boyce (fitness center, 167,066 sqft) saves approximately 12.9 kWh/hr at its peak deficit hour (16:00). Panther_education_Scarlett (classroom building, 29,469 sqft) shows a −2.0 kWh/hr midday deficit peaking at 12:00. This pattern is consistent with demand-side management practices: pre-cooling during morning off-peak hours, thermal storage utilization, or demand response enrollment that curtails HVAC during peak windows.

**Pattern 2 — Delayed Morning Start (Bear_assembly_Beatrice, Panther_education_Janis, Fox_assembly_Bradley):** These buildings begin their energy ramp-up substantially later in the morning than the model expects, minimizing pre-conditioning energy. Fox_assembly_Bradley (sports stadium, 809,530 sqft) saves approximately 8.6 kWh/hr at its peak deficit hour (07:00), suggesting the building management system leverages the building's enormous thermal mass effectively—beginning conditioning later than typical but maintaining comfort through stored heat. Bear_assembly_Beatrice (30,160 sqft) shows a similar pattern with −3.9 kWh/hr savings at 08:00. This is consistent with advanced BMS algorithms that incorporate weather forecasting and thermal mass estimation.

**Pattern 3 — Evening/Overnight Shutdown (Rat_office_Avis, Bear_education_Derek, Panther_office_Brent):** These buildings achieve aggressive load reduction after business hours. Rat_office_Avis (office, 474,680 sqft) saves approximately 40.6 kWh/hr at 23:00—the largest absolute savings among all candidates—reflecting rapid HVAC setback and lighting shutdown after evening occupancy ends. Bear_education_Derek (education, 68,146 sqft) shows −6.6 kWh/hr savings peaking at 23:00, consistent with strict after-hours energy policies. Panther_office_Brent (office, 10,743 sqft) extends this pattern into pre-dawn hours (23:00–06:00), suggesting deep overnight setback or near-complete HVAC shutdown.

**Deep dive — Fox_assembly_Boyce (Fitness Center, 167,066 sqft):**

This building illustrates the framework's diagnostic power. Its EUI Score is low (11.6, energy-intensive relative to CBECS public assembly median), yet NMBE = −2.7% (under-consuming relative to model expectations) with highly atypical patterns (Pattern Score = 35). ENERGY STAR would rate this building poorly based on annual EUI alone. Yet the hourly analysis reveals that the building suppresses afternoon peak consumption by 12.9 kWh/hr at its peak deficit hour (16:00)—creating a distinctive "afternoon dip" in its load profile that the model, trained on typical public assembly buildings, interprets as anomalous. This "anomalous" pattern is operationally desirable: the building is energy-intensive overall (inherent to fitness center operations), but manages *when* it consumes energy with apparent sophistication.

This building represents precisely the type that annual EUI-based frameworks miss but a temporal framework identifies: not efficient in aggregate terms, but specifically efficient during grid-critical hours, making it a candidate for best-practice documentation, demand response program enrollment, and cross-facility knowledge transfer.

---

## 6. Discussion

### 6.1 Theoretical Implications: Zero-Shot Error as Population-Referenced Atypicality

The central theoretical contribution of this paper is the reframing of zero-shot prediction error from an accuracy measure to a *population-referenced diagnostic signal*. This reframing rests on a key structural asymmetry between a population-trained model and a building-specific model.

A building-specific model (e.g., ARIMA fitted to a single building's history) learns whatever patterns that specific building exhibits; its prediction error reflects random fluctuations and measurement noise but not population-atypicality, because the model has no knowledge of what other buildings do. A population-trained model, by contrast, has learned the joint distribution of patterns across thousands of diverse buildings; its prediction error for a new building reflects the *distance* of that building's patterns from the learned population manifold.

This creates a diagnostic instrument: large zero-shot prediction error implies population-atypical patterns, irrespective of absolute consumption level. The model does not fail to predict high-EUI buildings systematically (r = −0.082 between EUI and CVRMSE, R² = 0.007); it fails to predict buildings whose *patterns* are unusual, regardless of their magnitude. This is precisely what we want from a temporal pattern consistency metric.

The parallel to ENERGY STAR is instructive. ENERGY STAR's WLS regression is also a population model—it predicts a building's expected EUI from its characteristics, and the prediction error (actual vs. predicted EUI) becomes the score. We are doing the same thing at the temporal level: the model predicts a building's expected hourly consumption from its own history and context, and the prediction error (CVRMSE) becomes the temporal pattern consistency score. The conceptual structure is identical; the temporal resolution is transformatively different.

This difference in temporal resolution has growing practical significance. As electricity grids incorporate higher shares of variable renewables and adopt time-differentiated pricing, two buildings with identical annual EUI can produce substantially different carbon emissions, electricity costs, and grid-level impacts depending on *when* they consume energy. A building that concentrates consumption during low-carbon overnight hours and a building that peaks during high-carbon afternoon hours are operationally equivalent under annual EUI benchmarking—but represent fundamentally different cases for grid decarbonization, demand response potential, and operating cost management. Existing annual benchmarking systems cannot distinguish between these cases; a framework that characterizes temporal consumption patterns can. This is the practical context that motivates moving beyond annual EUI toward the temporal pattern assessment developed in this paper—not as a replacement for aggregate efficiency metrics, but as a complementary diagnostic dimension that becomes increasingly necessary as grid carbon intensity and electricity pricing become more temporally heterogeneous.

### 6.2 The CVRMSE Decomposition: Resolving the CV Ambiguity

The CV-CVRMSE regression (CVRMSE = 0.541 × CV − 0.030, R² = 0.700) is a methodologically important finding that resolves the fundamental ambiguity in CV-based pattern assessment. By establishing the expected CVRMSE for a building given its inherent variability, we can determine whether the building's pattern is atypical *conditional on* its variability level—an assessment impossible with CV alone.

Consider two buildings, both with CV = 0.50 and CVRMSE = 0.35:
- **Building X:** Expected CVRMSE (from regression) = 0.541 × 0.50 − 0.030 = 0.241. Excess = 0.35 − 0.241 = 10.9 pp → ATYPICAL
- **Building Y:** Expected CVRMSE (from regression) = 0.241 (same). Excess = 0.35 − 0.241 = 10.9 pp → ATYPICAL

But now consider Building Z with CV = 0.60 and CVRMSE = 0.35:
- Expected CVRMSE = 0.541 × 0.60 − 0.030 = 0.295. Excess = 0.35 − 0.295 = 5.5 pp → Borderline ATYPICAL

And Building W with CV = 0.70 and CVRMSE = 0.35:
- Expected CVRMSE = 0.541 × 0.70 − 0.030 = 0.349. Excess = 0.35 − 0.349 = 0.1 pp → CV_DRIVEN

Buildings X/Y and Z have identical raw CVRMSE = 35% and similar CV values, but the decomposition reveals that only Buildings X/Y are genuinely anomalous; Building W's identical CVRMSE is fully explained by its higher inherent variability. This discrimination—invisible without the decomposition—is crucial for prioritizing operational investigations.

### 6.3 Robustness of the EUI-CVRMSE Independence Finding

A concern with our EUI Score computation is that it uses CBECS population-referenced z-scores without full ENERGY STAR covariate adjustment (operating hours, worker count, plug load intensity within building types). If the missing covariates correlate with CVRMSE, our near-independence finding (r = −0.082) might be an artifact of incomplete EUI adjustment.

We address this concern in two ways:

First, we note that the raw EUI-CVRMSE correlation (r = −0.082, p = 0.047, R² = 0.007) requires no EUI Score computation—it directly measures the relationship between annual energy intensity and temporal pattern deviation, independent of any scoring methodology. Although marginally significant at the 5% level, the effect size is negligible: EUI explains less than 1% of CVRMSE variance.

Second, we note that operating hours and worker count—the primary missing covariates—should primarily affect EUI level, not temporal pattern structure. A building that operates more hours has higher EUI; but whether its operational patterns during those hours are population-typical is a separate question that CVRMSE captures independently.

### 6.4 Addressing the Simulation-to-Reality Gap

A fundamental concern about using a simulation-trained model for real building diagnosis is the sim-to-real gap: systematic differences between simulated and real buildings may inflate CVRMSE across all real buildings, potentially confounding the population-referenced interpretation.

We address this concern empirically and conceptually. Empirically, parking garage buildings—which have the simplest, most predictable real-world patterns—achieve median CVRMSE of 7.7%, confirming the model achieves very low prediction error when real patterns are regular. Lodging buildings achieve median CVRMSE of 10.9%. These values, well below the 15% ASHRAE Guideline 14 acceptance threshold for M&V models, indicate the model is well-calibrated for regular real buildings—not systematically inflated.

Conceptually, the within-type z-score normalization for Pattern Score absorbs any systematic sim-to-real bias that operates at the building-type level. If all office buildings have CVRMSE elevated by X% due to sim-to-real differences in office building scheduling, the z-score removes this constant offset, and the Pattern Score reflects *within-type relative* atypicality—which is exactly what we want for benchmarking.

Furthermore, ENERGY STAR faces an analogous concern: CBECS survey data does not perfectly represent any individual building's context, and the WLS regression introduces prediction error that the system treats as building atypicality. The appropriate comparison is not between our framework and an idealized perfect predictor, but between our framework and existing benchmarking systems—all of which use population statistics as their reference.

### 6.5 Practical Deployment Pathway

The framework's minimal data requirement—hourly metered energy consumption + latitude/longitude—aligns naturally with smart meter infrastructure being deployed globally under mandatory benchmarking programs. In jurisdictions with building disclosure requirements (New York Local Law 97, California AB 802, EU Energy Performance of Buildings Directive), the required data is already collected annually or continuously.

The computational workflow is straightforward:
1. Collect 12+ months of hourly metered energy data per building
2. Retrieve geographic coordinates from building registry or permit records
3. Apply Box-Cox normalization per building; run zero-shot inference using pre-loaded model weights
4. Compute CVRMSE, NMBE, CV, and EUI (if floor area available) from stored predictions
5. Apply hierarchical scoring within peer groups (building type if known; all-buildings z-score otherwise)
6. Generate diagnostic report: quadrant, Level 2 classification, and (for ATYPICAL) NMBE direction + hourly profile

The entire workflow after data collection can run at portfolio scale in hours on standard computing infrastructure, enabling large-scale screening without building-specific expertise. This represents a substantial democratization of temporal building performance assessment relative to multi-KPI frameworks requiring expert KPI design and calibration.

### 6.6 Limitations and Boundary Conditions

**L1 — EUI score simplification.** Our CBECS population-referenced z-score approach does not adjust for operating hours, worker count, or plug load intensity within building types (unlike ENERGY STAR's regression-based adjustment). This may introduce noise in the EUI Score but does not affect the CVRMSE-based Pattern Score, and the core independence finding is robust (Section 6.3).

**L2 — Sample-specific decomposition regression.** The regression CVRMSE = 0.541 × CV − 0.030 (R² = 0.700) is fitted on BDG-2's 611 buildings across four North American sites. The coefficient α and intercept β will differ for other building populations, climate contexts, or meter types. The Excess CVRMSE threshold (5 pp) and the ATYPICAL classification boundary require recalibration on new datasets. Validation on independent datasets—ASHRAE GEPIII full dataset, urban energy disclosure datasets (New York, Chicago, Washington DC), European building portfolios—is the highest priority for future work.

**L3 — Low-load building filter.** Buildings with mean hourly load < 5 kWh/hr are excluded from ATYPICAL/NMBE analysis due to denominator inflation risk. Of BDG-2's 611 buildings, 23 fall below this threshold; for these, CVRMSE and NMBE metrics may be unreliable and results should be interpreted with caution or excluded from portfolio screening.

**L4 — Building type granularity.** BDG-2's "Education" category spans K-12 schools to research universities with substantially different operational profiles, HVAC systems, and occupancy patterns. Finer-grained taxonomy would improve within-type z-score normalization precision. Future work should explore sub-type clustering (e.g., using load shape clustering within nominal building types) to improve type-conditioning.

**L5 — Training data vintage and update.** Buildings-900K was generated using building codes and CBECS survey data from pre-2023 vintages. As building stock characteristics evolve (electrification, on-site renewables, EV charging integration), the model's learned distribution may drift from the current building population. Periodic model retraining on updated simulation corpora would maintain diagnostic validity.

**L6 — Behind-the-meter (BTM) resources and net metering ambiguity.** Buildings with on-site generation (photovoltaics), battery storage (ESS), geothermal systems, or active demand response participation create temporal load signatures—midday generation dips, charge-discharge cycling, seasonal heat pump patterns—that are indistinguishable from operational anomalies when only net meter data is available. The pre-trained model was trained on simulated consumption profiles without BTM resources; buildings with significant BTM may be classified as ATYPICAL due to their BTM-modified load shape rather than operational inefficiency.

This limitation parallels the fundamental challenge in measurement and verification (M&V): establishing a counterfactual baseline against which to measure performance when behind-the-meter conditions are unknown and evolving. It also explains the enduring appeal of simplified annual benchmarking (e.g., ENERGY STAR): by aggregating to annual resolution, BTM temporal signatures are absorbed into a single EUI value, avoiding the interpretive complexity of hourly patterns.

However, three considerations mitigate this concern for the present study:

1. **BDG-2 temporal context.** The BDG-2 dataset comprises 2016–2017 data from U.S. university campuses. Commercial-scale BTM deployment accelerated primarily after 2020 (IEA 2023); significant BTM penetration in BDG-2 buildings is unlikely, though not verifiable from meter data alone.

2. **BTM affects annual EUI equally.** ENERGY STAR uses net energy consumption; a building with rooftop solar achieves lower net EUI regardless of operational efficiency. Annual benchmarking does not solve the BTM problem—it merely conceals it. Hourly-resolution analysis can at least *detect* characteristic BTM signatures (e.g., midday dips consistent with solar generation), providing diagnostic information invisible to annual metrics.

3. **Screening vs. diagnosis.** Our framework identifies buildings whose temporal patterns deviate from population-representative norms—it does not claim to diagnose the *cause* of that deviation. Whether atypicality stems from operational inefficiency, BTM resources, or equipment faults, the building warrants investigation. The framework's value lies in prioritizing which buildings to investigate, not in replacing on-site assessment.

For future deployment in high-BTM-penetration contexts, we recommend: (a) using gross meter data (before net metering) where available through advanced metering infrastructure; (b) incorporating utility interconnection records or distributed energy resource registries to flag known BTM installations; and (c) developing BTM-aware signature libraries to distinguish BTM-driven patterns from operational anomalies.

---

## 7. Conclusions

This paper presented a hierarchical, three-level building energy performance evaluation framework that integrates EUI-based efficiency assessment with zero-shot prediction CVRMSE from a large-scale pre-trained load forecasting model as a temporal pattern consistency metric. Applied to 611 real buildings from the Building Data Genome Project 2 (BDG-2) encompassing 9,247,992 observation-prediction timestep pairs, the framework yielded the following principal findings:

**Finding 1:** EUI and zero-shot CVRMSE are empirically near-independent (r = −0.029, p = 0.48 for all 611 buildings; r = −0.082, p = 0.047, R² = 0.007 for 583 CBECS-mapped buildings; r = −0.291, R² = 0.085 for normalized scores). Neither metric is a proxy for the other; both must be measured for a complete building performance assessment. The independence is theoretically expected—EUI captures aggregate annual consumption intensity while CVRMSE captures the temporal structure of energy use pattern relative to a population-calibrated reference.

**Finding 2:** Benchmarked against CBECS 2018 Table C14 electricity-only median EUI thresholds, the hierarchical four-quadrant framework (Level 1) classifies 583 CBECS-mapped BDG-2 buildings as: Excellent (A: 122, 20.9%), Efficient but Irregular (B: 107, 18.4%), Consistent but Inefficient (C: 250, 42.9%), and Needs Improvement (D: 104, 17.8%). The framework correctly directs buildings in Quadrant C (consistent, inefficient) toward capital investment rather than operational changes—a distinction invisible to EUI-alone benchmarking. The dominance of Quadrant C reflects BDG-2's university campus characteristics (research facilities, 24/7 operations, laboratory equipment) which systematically exceed national commercial building medians.

**Finding 3:** The CVRMSE decomposition (Level 2) reveals that 70% of cross-building CVRMSE variance is explained by inherent load variability (CV), and only 58 buildings (9.9% of 583 CBECS-mapped) are genuinely ATYPICAL—exhibiting pattern deviations exceeding what their inherent variability would predict. The large majority of high-CVRMSE buildings (CV_DRIVEN, 153 buildings) have inherently complex consumption patterns consistent with their use type, making operational intervention less immediately actionable.

**Finding 4:** NMBE directional analysis (Level 3) is statistically meaningful only for ATYPICAL buildings (Mann-Whitney U, p < 0.0001). For NORMAL and CV_DRIVEN buildings, model errors are symmetric and near zero; applying NMBE guidance to these buildings would generate misleading recommendations. This finding motivates the hierarchical restriction of Level 3 analysis to ATYPICAL buildings only.

**Finding 5:** 64.7% of ENERGY STAR-certifiable buildings (EUI Score ≥ 75, n=85) exhibit temporal pattern irregularities (Pattern Score < 50) undetectable by annual EUI benchmarking. Conversely, 42.9% of buildings (Quadrant C, n=250) with EUI Score < 50 (exceeding CBECS 2018 C14 median) exhibit highly consistent temporal patterns (Pattern Score ≥ 50), indicating their efficiency gap is structural rather than operational. Across all four reversal case types, buildings receive substantively different assessments when the temporal dimension is incorporated beyond annual EUI alone.

**Finding 6:** Nine best-practice candidate buildings are identified through systematic under-consumption (NMBE < −2%) combined with genuine pattern atypicality, with three recurring operational patterns: midday/afternoon peak suppression, delayed morning start, and evening/overnight shutdown. These patterns represent population-referenced operational benchmarks with direct implications for peer buildings in the same type and climate category.

**Finding 7:** The entire temporal diagnostic dimension of the framework requires only hourly metered energy consumption and geographic coordinates—no floor area, building type, occupant counts, or operating schedules. This minimal data requirement enables deployment at portfolio scale from smart meter infrastructure alone.

The broader implication of this work is that the building energy benchmarking field has access to a powerful new diagnostic instrument that was previously invisible: the temporal structure of the gap between a pre-trained model's expectations and a building's actual behavior. This gap is not model error to be minimized—it is building-level diagnostic signal to be interpreted. By reframing zero-shot prediction error as population-referenced pattern atypicality, and by developing a statistically validated hierarchical framework for interpreting this signal, we have demonstrated that it is possible to move substantially beyond "how much energy does this building use?" toward "how does this building use energy, and what should be done about it?"

As global building performance standards increasingly incorporate temporal dimensions—in demand response programs, time-of-use tariffs, carbon-intensity-weighted energy metrics, and smart grid integration—the analytical capability developed in this framework will become not merely a supplement to existing benchmarking but a core diagnostic requirement. We acknowledge that the proliferation of behind-the-meter (BTM) resources (solar PV, battery storage, geothermal) complicates temporal pattern interpretation, as BTM-modified load shapes may be indistinguishable from operational anomalies using net meter data alone (Section 6.6, L6). However, this challenge equally affects annual EUI benchmarking, which conceals rather than resolves BTM effects. Hourly-resolution analysis uniquely enables BTM signature detection and, with gross metering or BTM registry integration, can disentangle generation effects from consumption patterns. We recommend that future building performance certification programs adopt temporal pattern consistency—whether measured via the present CVRMSE-based approach or a refined successor—as a mandatory second dimension alongside aggregate EUI, with appropriate BTM flagging protocols, enabling a genuinely comprehensive, actionable, and data-driven characterization of the built environment's energy performance.

---

## CRediT Author Statement

[Author A]: Conceptualization, Methodology, Software, Formal analysis, Writing – Original Draft
[Author B]: Conceptualization, Supervision, Writing – Review & Editing, Funding acquisition
[Author C]: Data curation, Validation, Writing – Review & Editing

---

## Declaration of Competing Interest

The authors declare no competing financial interests.

---

## Data Availability

- **BDG-2 dataset:** Publicly available at https://github.com/buds-lab/building-data-genome-project-2
- **BuildingsBench model and weights:** Publicly available at https://github.com/NREL/BuildingsBench
- **Analysis code and results:** Will be made available upon acceptance at [repository URL]

---

## Acknowledgments

The authors acknowledge the National Renewable Energy Laboratory (NREL) for developing and releasing the BuildingsBench framework, model weights, and Buildings-900K simulation dataset under open-source license. We acknowledge the Building Data Genome Project team for curating and releasing the BDG-2 dataset. We also acknowledge the U.S. Energy Information Administration for the CBECS survey data that underpins the Buildings-900K simulation corpus.

---

## References

[1] United Nations Environment Programme (UNEP) & Global Alliance for Buildings and Construction (GlobalABC). 2023 Global Status Report for Buildings and Construction: Beyond Foundations. Nairobi: UNEP, 2023. https://doi.org/10.59117/20.500.11822/45095

[2] United Nations Environment Programme (UNEP). 2022 Global Status Report for Buildings and Construction. Nairobi: UNEP, 2022. https://globalabc.org/resources/publications/2022-global-status-report-buildings-and-construction

[3] Pérez-Lombard, L., Ortiz, J., & Pout, C. (2008). A review on buildings energy consumption information. Energy and Buildings, 40(3), 394–398. https://doi.org/10.1016/j.enbuild.2007.03.007

[4] U.S. Environmental Protection Agency (EPA). ENERGY STAR Score for Commercial Buildings in the United States. Technical Reference. Washington, DC: EPA, 2023. https://www.energystar.gov/sites/default/files/buildings/tools/ENERGY_STAR_Score_Technical_Reference.pdf

[5] Arjunan, P., Miller, C., & Poolla, K. (2020). EnergyStar++: Towards more accurate and explanatory building energy benchmarking. Applied Energy, 276, 115413. https://doi.org/10.1016/j.apenergy.2020.115413

[6] Institute for Market Transformation (IMT). Benchmark 8760: The Case for Hourly Benchmarking of Commercial Buildings. Washington, DC: IMT, 2022. https://www.imt.org/resources/benchmark-8760/

[7] Piscitelli, M. S., Chiara, F., & Capozzoli, A. (2019). Recognition and classification of typical load profiles in buildings with non-intrusive learning approaches. Applied Energy, 255, 113727. https://doi.org/10.1016/j.apenergy.2019.113727

[8] Li, X., Yao, R., Li, Q., Ding, Y., & Li, B. (2018). An object-oriented energy benchmark for the evaluation of the office building stock. Utilities Policy, 51, 1–11. https://doi.org/10.1016/j.jup.2018.01.008

[9] Goswami, M., Szafer, K., Choudhry, A., Cai, Y., Li, S., & Dubrawski, A. (2024). MOMENT: A family of open time-series foundation models. In Proceedings of the 41st International Conference on Machine Learning (ICML 2024), Vienna, Austria.

[10] Ansari, A. F., Stella, L., Turkmen, C., Zhang, X., Mercado, P., Shen, H., Metaxas, D., Lim, B., de Bayser, M., Bengio, S., & Wang, Y. (2024). Chronos: Learning the language of time series. Transactions on Machine Learning Research, ISSN 2835-8856.

[11] Das, A., Kong, W., Sen, R., & Zhou, Y. (2024). A decoder-only foundation model for time-series forecasting. In Proceedings of the 41st International Conference on Machine Learning (ICML 2024), Vienna, Austria.

[12] Emami, P., Adams, S., Bhaskaran, S., He, T., & Lunacek, M. (2023). BuildingsBench: A large-scale dataset of 900K buildings and benchmark for short-term load forecasting. In Advances in Neural Information Processing Systems (NeurIPS 2023), 36. https://doi.org/10.48550/arXiv.2307.00142

[13] Bordass, B., Cohen, R., & Field, J. (2004). Energy performance of non-domestic buildings: Closing the credibility gap. In Proceedings of the International Conference on Improving Energy Efficiency in Commercial Buildings (IEECB'04), Frankfurt, Germany.

[14] Scofield, J. H. (2009). Do LEED-certified buildings save energy? Not really. Energy and Buildings, 41(12), 1386–1390. https://doi.org/10.1016/j.enbuild.2009.08.006

[15] American Society of Heating, Refrigerating and Air-Conditioning Engineers (ASHRAE). Building EQ: A Tool for Building Energy Audits and Benchmarking. Atlanta, GA: ASHRAE, 2014.

[16] European Commission. Commission Recommendation (EU) 2019/786 of 8 May 2019 on Building Renovation. Official Journal of the European Union, L 127/34, 2019.

[17] Park, J. Y., Yang, X., Miller, C., Arjunan, P., & Nagy, Z. (2019). Apples or oranges? Identification of fundamental load shape profiles for benchmarking buildings using a large and diverse dataset. Applied Energy, 236, 1280–1295. https://doi.org/10.1016/j.apenergy.2018.12.025

[18] Andrews, A., & Jain, R. K. (2022). Beyond energy efficiency: A clustering approach to embed demand flexibility into building energy benchmarking. Applied Energy, 327, 119989. https://doi.org/10.1016/j.apenergy.2022.119989

[19] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). Attention is all you need. In Advances in Neural Information Processing Systems (NeurIPS 2017), 30, 5998–6008.

[20] Miller, C., Kathirgamanathan, A., Picchetti, B., Arjunan, P., Park, J. Y., Nagy, Z., Raftery, P., Hobson, B. W., Shi, Z., & Meggers, F. (2020). The building data genome project 2, energy meter data from the ASHRAE great energy predictor III competition. Scientific Data, 7(1), 368. https://doi.org/10.1038/s41597-020-00712-x

[21] U.S. Energy Information Administration (EIA). 2018 Commercial Buildings Energy Consumption Survey (CBECS): Building Characteristics. Washington, DC: EIA, 2022. https://www.eia.gov/consumption/commercial/

[22] Granderson, J., Price, P. N., Jump, D., Addy, N., & Sohn, M. D. (2015). Automated measurement and verification: Performance of public domain whole-building electric baseline models. Applied Energy, 144, 106–113. https://doi.org/10.1016/j.apenergy.2015.01.054

[23] American Society of Heating, Refrigerating and Air-Conditioning Engineers (ASHRAE). ASHRAE Guideline 14-2014: Measurement of Energy, Demand, and Water Savings. Atlanta, GA: ASHRAE, 2014.

[24] U.S. Energy Information Administration (EIA). 2018 Commercial Buildings Energy Consumption Survey (CBECS): Consumption and Expenditures Tables. Washington, DC: EIA, 2022. https://www.eia.gov/consumption/commercial/data/2018/

[25] Katipamula, S., & Brambley, M. R. (2005). Methods for fault detection, diagnostics, and prognostics for building systems—A review, Part I. HVAC&R Research, 11(1), 3–25. https://doi.org/10.1080/10789669.2005.10391123

[26] Zhao, Y., Li, T., Zhang, X., & Zhang, C. (2019). Artificial intelligence-based fault detection and diagnosis methods for building energy systems: Advantages, challenges and the future. Renewable and Sustainable Energy Reviews, 109, 85–101. https://doi.org/10.1016/j.rser.2019.04.021

---

## Appendix A: Sensitivity Analysis — EUI Score Calculation Method

To verify the robustness of the EUI-CVRMSE independence finding to EUI calculation methodology, we repeated the correlation analysis using alternative EUI scoring approaches:

| EUI Scoring Method | EUI ↔ CVRMSE (raw) | EUI Score ↔ Pattern Score |
|--------------------|-------------------|--------------------------|
| CBECS C14 z-score (our method, n=583) | r = −0.082 (R² = 0.007) | r = −0.291 |
| Within-type percentile (BDG-2 relative ranking) | r = −0.082 (R² = 0.007) | r = −0.276 |
| All-buildings percentile (no type adjustment) | r = −0.082 (R² = 0.007) | r = −0.278 |
| Raw EUI (no normalization) | r = −0.082 (R² = 0.007) | – |
| Log-transformed EUI percentile | r = −0.082 (R² = 0.007) | r = −0.278 |

**Note:** The raw EUI ↔ CVRMSE correlation is invariant across EUI scoring methods because it uses unscored kBtu/sqft vs. CVRMSE. On the full 611-building sample (including Parking, Other, Technology, Utility), r = −0.029 (p = 0.48); on the 583 CBECS-mapped subset used throughout this paper, r = −0.082 (p = 0.047). Although marginally significant, the effect size is negligible (R² = 0.007). The EUI Score ↔ Pattern Score correlation varies modestly (r ∈ [−0.291, −0.276]) but consistently indicates weak negative association explaining 7.6–8.5% of variance. These results confirm the near-independence finding is robust to scoring methodology.

---

## Appendix B: Complete Quadrant × Level 2 × Level 3 Classification Table

The complete per-building classification results are provided in the supplementary data file `hierarchical_scores_complete.csv`, containing the following fields for each of the 611 buildings: building_id, site, building_type, floor_area_sqft, EUI_kbtu_sqft, EUI_Score, CV, CVRMSE, NMBE, Expected_CVRMSE, Excess_CVRMSE, Pattern_Score, Quadrant (A/B/C/D), Level2_Class (NORMAL/CV_DRIVEN/ATYPICAL), Level3_Class (OVER_CONSUMING/NEUTRAL/UNDER_CONSUMING, ATYPICAL only), and flag_best_practice_candidate.

---

## Appendix C: Diagnostic Report Template

The following template illustrates the standardized diagnostic report generated for each building:

```
═══════════════════════════════════════════════════════════════
BUILDING ENERGY DIAGNOSTIC REPORT
Generated: [Date] | Model: TransformerWithGaussian-L (zero-shot)
═══════════════════════════════════════════════════════════════

Building ID   : [ID]           Site   : [Site]
Type          : [Type]         Sqft   : [Floor Area]
Location      : [Lat/Lon]      Period : [Start] – [End]

───────────────────────────────────────────────────────────────
LEVEL 1: EFFICIENCY × PATTERN ASSESSMENT
───────────────────────────────────────────────────────────────
  Annual EUI       : [X.X] kWh/sqft/yr
  EUI Score        : [XX] / 100  (CBECS population-referenced, 50 = national median)

  CVRMSE           : [XX.X]%
  Pattern Score    : [XX] / 100  (BDG-2 referenced, 50 = mean)

  ★ QUADRANT       : [A/B/C/D] — [Description]

  Primary Recommendation:
  → [Maintain operations / Investigate operational controls /
     Capital investment (systems/envelope) / Operational + capital review]

───────────────────────────────────────────────────────────────
LEVEL 2: PATTERN CAUSE DECOMPOSITION  [Quadrant B or D only]
───────────────────────────────────────────────────────────────
  Inherent Variability (CV)   : [X.XXX]
  Expected CVRMSE (from CV)   : [XX.X]%
  Actual CVRMSE               : [XX.X]%
  Excess CVRMSE               : [±X.X] pp

  ★ CLASSIFICATION : [CV_DRIVEN / ATYPICAL]

  Interpretation:
  → [High CVRMSE primarily reflects inherent operational complexity.
     Operational changes likely have limited impact on pattern consistency.
     /
     Pattern deviation exceeds what inherent variability predicts.
     Genuine operational anomaly detected. Proceed to Level 3.]

───────────────────────────────────────────────────────────────
LEVEL 3: CONSUMPTION DIRECTION ANALYSIS  [ATYPICAL only]
───────────────────────────────────────────────────────────────
  NMBE                        : [±X.X]%

  ★ DIRECTION      : [OVER-CONSUMING / NEUTRAL / UNDER-CONSUMING]

  Hourly Bias Profile (top 3 hours by |bias|):
    [HH:00]  →  [±X.X kWh] average vs. model expectation
    [HH:00]  →  [±X.X kWh] average vs. model expectation
    [HH:00]  →  [±X.X kWh] average vs. model expectation

  Actionable Recommendation:
  → [Specific operational intervention based on direction and hourly profile]
═══════════════════════════════════════════════════════════════
```

---

## Appendix D: Causal Decomposition of High-CVRMSE Buildings

**Table D1. Causal decomposition of high-CVRMSE buildings (CVRMSE > 20%, n = 185)**

| Cause | Count | % of High-CVRMSE | Description |
|-------|-------|-----------------|-------------|
| HIGH_CV (CV above type median) | 162 | 87.6% | High inherent variability drives CVRMSE |
| LOW_DENOM (mean < 25th percentile) | 88 | 47.6% | Small mean load inflates normalized CVRMSE |
| ATYPICAL (Excess > 5 pp) | 62 | 33.5% | Genuine pattern deviation from population model |
| CV_DRIVEN only (single cause) | 66 | 35.7% | CVRMSE fully explained by CV alone |

Note: Categories are not mutually exclusive; a building may appear in multiple rows.

---

## Appendix E: Quadrant Distribution by Building Type

**Table E1. Quadrant distribution by building type (CBECS 2018 C14, selected types, n ≥ 15)**

| Building Type | n | A: Excellent (%) | B: Eff. but Irreg. (%) | C: Cons. Ineffic. (%) | D: Needs Impr. (%) |
|---------------|---|:----------------:|:----------------------:|:---------------------:|:------------------:|
| Lodging | 54 | 55.6 | 16.7 | 18.5 | 9.3 |
| Public Services | 98 | 24.5 | 22.4 | 34.7 | 18.4 |
| Office | 71 | 18.3 | 11.3 | 42.3 | 28.2 |
| Education | 254 | 17.3 | 22.8 | 44.9 | 15.0 |
| Public Assembly | 80 | 10.0 | 7.5 | 62.5 | 20.0 |

---

*Manuscript word count (main text): approximately 10,500 words*
*Number of tables: 10 (main) + 2 (appendix)*
*Number of figures: 4*
*Supplementary materials: 1 data file (hierarchical_scores_complete.csv)*

*Prepared: March 2026*
