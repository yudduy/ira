# Project: IRA Corporate Messaging Stability Analysis

This project analyzes changes in corporate messaging around the passage of the U.S. Inflation Reduction Act (IRA) to validate the use of contemporary company descriptions for academic research.

---
### 1. Core Research Objective

**First Principle**: Research conclusions are only as reliable as the underlying data. If a major policy intervention (the IRA) caused firms to systematically alter their self-descriptions, using post-policy descriptions as a proxy for a pre-policy state is methodologically unsound.

**Objective**: To conduct a **Stability Validation Study**. We test the hypothesis that corporate messaging for a sample of firms remained largely stable before and after the IRA's passage in August 2022. This validates the use of 2025-era data as representative of the firms' 2022 strategic focus.

---
### 2. Methodology

Our methodology is designed for rigor and resilience, adapting to the realities of archival web data.

#### **Data Sourcing & Scoping: Annual Windows & Page Targeting**
* **Method**: We compare website snapshots from the **full year of 2022** (pre-IRA baseline) against the **full year of 2023** (post-IRA implementation). This annual window maximizes the probability of finding usable data for each company, a standard practice in archival studies to ensure statistical power where data is sparse.
* **Rationale**: To ensure **construct validity**, our search is not limited to homepages. The analysis targets a list of strategically relevant pages (`/about`, `/products`, `/sustainability`, `/esg`, etc.) where a company's core identity is most clearly articulated.

#### **Analytical Framework: Hierarchical Assessment**
**First Principle**: A firm's public messaging is a multi-dimensional strategic signal. To understand a policy's impact, we must measure change across these dimensions.

Our Large Language Model (LLM) is prompted to deconstruct "change" into a hierarchical, theory-driven framework, ensuring an auditable and transparent reasoning process:

1.  **Lexical Change (The *What*)**: Measures objective changes in vocabulary and keywords, providing a direct signal of a messaging update.
2.  **Strategic Framing (The *Why*)**: Assesses shifts in the company's **narrative identity** or mission—a deeper change than keywords alone.
3.  **Target Audience (The *Who*)**: Analyzes communication style to detect shifts in stakeholder focus (e.g., from customers to investors), testing the "strategic signaling" axiom of management science.
4.  **IRA Alignment (The *Causal Link*)**: Conducts a sophisticated search for explicit policy mentions, related tax codes (`45Q`, `ITC`), and conceptual language ("domestic content") to measure both implicit and explicit policy alignment.

The LLM first assesses these granular components and then synthesizes an `overall_assessment`, from "none" to "major," providing a final, justified conclusion for each company.

---
### 3. Technical Implementation

**First Principle**: A research tool must be resilient to real-world conditions and produce transparent, auditable data.

* **Architecture**: The Python-based tool is modular, separating data loading, web scraping, and AI analysis into distinct components for clarity and maintainability.
* **Tooling**: We use `requests` and `BeautifulSoup` for efficient data extraction from the Wayback Machine's static HTML archives.
* **Resilience**: The tool implements a **retry mechanism with exponential backoff** and robust timeouts to handle API rate limits and network latency.
* **Transparency**: The final output is a single, flattened CSV table containing all granular assessments from the LLM, ensuring full traceability from raw text to the final conclusion.
* **Handling Missing Data**: We acknowledge that a 100% snapshot success rate is impossible. The strategy is to **document the success rate** and perform an **attrition analysis** to check for selection bias (e.g., are smaller firms more likely to be missing?), ensuring the limitations of the dataset are transparently reported.

---
### 4. How to Run

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd ira_corporate_analysis
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up your environment:**
    Create a `.env` file in the root directory and add your OpenAI API key:
    ```
    OPENAI_API_KEY="your_api_key_here"
    ```
    Alternatively, the script can read the key from Google Colab secrets.

4.  **Execute the analysis:**
    ```bash
    python main.py --csv path/to/your/data.csv --sample 50 --output results.csv
    ```
    * `--csv`: (Required) Path to the input PitchBook CSV file.
    * `--sample`: (Optional) Number of companies to sample for a test run.
    * `--output`: (Optional) Path for the output results file.

---
### 5. Project Structure

```
ira_corporate_analysis/
├── .gitignore
├── README.md
├── requirements.txt
├── config.py              # Stores all constants and configuration settings
├── main.py                # Main entry point for running the analysis
└── src/
    ├── __init__.py
    ├── analyzer.py          # Core orchestrator class
    ├── data_processing.py   # Handles loading and cleaning input data
    ├── llm_handler.py       # Manages interaction with the OpenAI API
    └── wayback_client.py    # Manages interaction with the Wayback Machine
```

---
### 6. Output Format

The analysis produces a comprehensive CSV file with the following key columns:

* **Company Information**: `company_name`, `domain`, `website`
* **Analysis Status**: `status` (completed/insufficient_snapshots/content_extraction_failed/analysis_error)
* **Snapshot URLs**: `pre_ira_snapshot_url`, `post_ira_snapshot_url`
* **Content Metrics**: `pre_word_count`, `post_word_count`
* **Hierarchical Analysis Results**:
  - `lexical_has_changed`, `lexical_summary`
  - `framing_has_changed`, `framing_from_narrative`, `framing_to_narrative`, `framing_summary`
  - `audience_has_changed`, `audience_primary_audience`, `audience_summary`
  - `ira_alignment_detected`, `ira_evidence_type`, `ira_specific_evidence`, `ira_reasoning`
  - `overall_change_level`, `overall_confidence`, `overall_synthesis_reasoning`

---
### 7. Research Applications

This tool enables researchers to:

* **Validate Data Stability**: Test whether corporate messaging remained stable around policy interventions
* **Policy Impact Studies**: Measure how major legislation affects corporate communications
* **Event Study Methodology**: Apply established academic frameworks to web archival data
* **Longitudinal Analysis**: Track changes in corporate identity and strategic positioning over time

---
### 8. Limitations and Considerations

* **Data Availability**: Not all companies will have complete archival data for the target periods
* **Snapshot Quality**: Archived content may be incomplete or contain Wayback Machine artifacts
* **LLM Interpretation**: Analysis depends on the language model's ability to detect nuanced changes
* **Rate Limiting**: Wayback Machine API has usage limits that may slow large-scale analysis

---
### 9. Citation

If you use this tool in your research, please cite:

```
IRA Corporate Messaging Stability Analysis Tool
Academic Research - Stanford Graduate School of Business
Version 5.0 (Hierarchical Analysis Update)
```

---
### 10. Contributing

This is an academic research tool. For questions, issues, or contributions, please contact the research team or open an issue in this repository.

---

*Last updated: September 2025*
