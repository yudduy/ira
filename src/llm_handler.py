# src/llm_handler.py
"""
Handles hierarchical content analysis using the OpenAI API.
"""
import json
import logging
from typing import Dict
import openai
import config

logger = logging.getLogger(__name__)

class LLMHandler:
    """Manages the analysis of text content using an LLM."""

    def __init__(self, client: openai.AsyncOpenAI):
        self.client = client

    async def analyze_content_change(self, pre_content: str, post_content: str) -> Dict:
        """
        Analyzes content changes using a hierarchical, multi-construct prompting strategy.
        """
        system_prompt = (
            "You are a research analyst specializing in corporate communications, strategic "
            "management, and policy analysis. Your task is to rigorously analyze and compare "
            "two versions of a company's website text: one from before the US Inflation "
            "Reduction Act (IRA) and one from after."
        )

        user_prompt = f"""
        Compare the PRE-IRA and POST-IRA texts below. Analyze the changes along three specific dimensions first (lexical, framing, audience), then assess IRA alignment, and finally synthesize these findings into an overall assessment.

        ## PRE-IRA TEXT (Full Year 2022):
        {pre_content[:config.PROMPT_CONTENT_LIMIT]}

        ## POST-IRA TEXT (Full Year 2023):
        {post_content[:config.PROMPT_CONTENT_LIMIT]}

        Respond with ONLY a JSON object following this exact structure:
        {{
          "change_analysis": {{
            "lexical_change": {{
              "has_changed": true,
              "summary": "Briefly describe the key changes in vocabulary and keywords."
            }},
            "strategic_framing": {{
              "has_changed": false,
              "from_narrative": "Describe the company's core identity/mission in the pre-IRA text.",
              "to_narrative": "Describe the company's core identity/mission in the post-IRA text.",
              "summary": "Summarize the change in the company's narrative identity. Note 'No significant change' if applicable."
            }},
            "target_audience": {{
              "has_changed": false,
              "primary_audience": "Describe the main audience addressed in both texts (e.g., Consumers, B2B Customers, Investors, Policymakers).",
              "summary": "Summarize any shift in the target audience. Note 'No significant change' if applicable."
            }}
          }},
          "ira_alignment": {{
             "alignment_detected": false,
             "evidence_type": "none|explicit_mention|tax_code|conceptual_language",
             "specific_evidence": ["List specific terms like 'Inflation Reduction Act', '45Q', 'ITC', 'domestic content' if found."],
             "reasoning": "Provide your reasoning for the alignment assessment."
          }},
          "overall_assessment": {{
            "change_level": "none|minor|moderate|major",
            "confidence": 0.0,
            "synthesis_reasoning": "Synthesize your findings from the analyses above to justify the overall change_level."
          }}
        }}
        """

        try:
            response = await self.client.chat.completions.create(
                model=config.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=config.OPENAI_TEMPERATURE,
                max_tokens=config.OPENAI_MAX_TOKENS
            )
            analysis_json = json.loads(response.choices[0].message.content)
            return {'success': True, 'data': analysis_json}
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            return {'success': False, 'error': str(e)}
