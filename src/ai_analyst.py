# src/ai_analyst.py
import pandas as pd
from typing import Optional, List
import logging

# Configure logging
logger = logging.getLogger(__name__)

class AIAnalyst:
    """
    A unified interface for AI analysis using Google Gemini (default), Anthropic, or OpenAI.
    """
    
    def __init__(self, api_key: str, provider: str = "gemini", model: Optional[str] = None):
        self.api_key = api_key
        self.provider = provider.lower()
        self.model = model
        self.client = self._initialize_client()

    def _initialize_client(self):
        """Initializes the appropriate client based on the selected provider."""
        if not self.api_key:
            return None

        try:
            if self.provider == "gemini":
                from google import genai
                # Use a specific, stable model version to avoid 404 errors
                # gemini-1.5-flash-002 is the updated stable flash model
                self.model = self.model or "gemini-1.5-flash-002" 
                return genai.Client(api_key=self.api_key)
            
            elif self.provider == "anthropic":
                from anthropic import Anthropic
                # Default to Claude 3.5 Sonnet
                self.model = self.model or "claude-3-5-sonnet-20241022"
                return Anthropic(api_key=self.api_key)
            
            elif self.provider == "openai":
                from openai import OpenAI
                # Default to GPT-4o
                self.model = self.model or "gpt-4o"
                return OpenAI(api_key=self.api_key)
            
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
                
        except ImportError as e:
            raise ImportError(f"Missing library for {self.provider}. Please install it (e.g., `pip install google-genai`). Error: {e}")

    def _format_dataframe_for_prompt(self, df: pd.DataFrame, title: str) -> str:
        """Helper to convert DF to a clean markdown string for the LLM."""
        if df is None or df.empty:
            return f"{title}: No data available."
        return f"{title}:\n{df.to_markdown(index=True, floatfmt='.2f')}"

    def analyze_factor(self, 
                       factor_name: str, 
                       stats_df: pd.DataFrame, 
                       turnover_df: pd.DataFrame,
                       correlation_df: Optional[pd.DataFrame] = None) -> str:
        """
        Generates a quantitative assessment using the configured provider.
        """
        if not self.client:
            return "⚠️ **Error:** API Key is missing. Please enter your API Key in the sidebar."

        # 1. Construct Data Context
        stats_str = self._format_dataframe_for_prompt(stats_df, "Factor Portfolio Statistics")
        turnover_str = self._format_dataframe_for_prompt(turnover_df, "Portfolio Turnover")
        
        corr_context = ""
        if correlation_df is not None and not correlation_df.empty:
            if factor_name in correlation_df.columns:
                relevant_corr = correlation_df[[factor_name]].sort_values(by=factor_name, ascending=False)
                corr_context = self._format_dataframe_for_prompt(relevant_corr, "Correlations")

        # 2. Build Prompts
        system_instruction = (
            "You are a Senior Quantitative Researcher. Interpret the backtest data for the equity factor. "
            "Focus on Sharpe ratio, tail risk (Skewness), and turnover. Be concise and professional."
        )

        user_prompt = f"""
        Analyze the **{factor_name}** factor based on this data:

        {stats_str}
        {turnover_str}
        {corr_context}

        Provide a 3-part assessment:
        1. **Performance**: Drivers of alpha (Longs vs Shorts).
        2. **Risk**: Sharpe, Skewness, Drawdowns.
        3. **Implementation**: Turnover and diversification benefits.
        """

        # 3. Call the specific Provider API
        try:
            if self.provider == "gemini":
                return self._call_gemini(system_instruction, user_prompt)
            elif self.provider == "anthropic":
                return self._call_anthropic(system_instruction, user_prompt)
            elif self.provider == "openai":
                return self._call_openai(system_instruction, user_prompt)
            
        except Exception as e:
            return self._format_error(e)

    def _call_gemini(self, sys_prompt, user_prompt):
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=user_prompt,
                config={
                    'system_instruction': sys_prompt,
                    'temperature': 0.3
                }
            )
            return response.text
        except Exception as e:
            # Enhanced error handling for Gemini 404s
            error_str = str(e)
            if "404" in error_str and "models/" in error_str:
                available_models = self._list_gemini_models()
                model_list_str = "\n- ".join(available_models[:5]) # Show top 5
                raise Exception(
                    f"Model '{self.model}' not found (404). \n"
                    f"Try one of these valid models:\n- {model_list_str}\n\n"
                    f"Original Error: {e}"
                )
            raise e

    def _list_gemini_models(self) -> List[str]:
        """Helper to list available Gemini models for debugging."""
        try:
            models = self.client.models.list()
            # Filter for generateContent supported models
            return [m.name.split('/')[-1] for m in models if 'generateContent' in m.supported_generation_methods]
        except:
            return ["gemini-1.5-flash-002", "gemini-1.5-pro-002", "gemini-2.0-flash-exp"]

    def _call_anthropic(self, sys_prompt, user_prompt):
        message = self.client.messages.create(
            model=self.model,
            max_tokens=1000,
            temperature=0.3,
            system=sys_prompt,
            messages=[
                {"role": "user", "content": user_prompt}
            ]
        )
        return message.content[0].text

    def _call_openai(self, sys_prompt, user_prompt):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content

    def _format_error(self, e: Exception) -> str:
        """Standardized error formatting."""
        err_str = str(e).lower()
        
        # Quota Errors
        if "429" in err_str or "quota" in err_str or "exhausted" in err_str:
            return f"### ⚠️ Quota Exceeded ({self.provider.title()})\nYou have hit your rate limit or quota. Check your {self.provider.title()} billing settings."
        
        # Auth Errors
        if "401" in err_str or "invalid api key" in err_str or "403" in err_str:
            return f"### ⚠️ Authentication Failed\nYour {self.provider.title()} API Key appears invalid."
            
        # Model Not Found (Specific 404 handling)
        if "404" in err_str and "not found" in err_str:
             return f"### ⚠️ Model Not Found\nThe model `{self.model}` is not available in your region or API version.\n\n**Details:** {str(e)}"

        return f"### ⚠️ Analysis Error\nAn unexpected error occurred: `{str(e)}`"