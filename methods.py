import asyncio
import datetime
from typing import List, Dict, Optional
import json
from anthropic import Anthropic, AsyncAnthropic
import re

from eval_utils.anthropic_model import (
    get_anthropic_client_sync,
    get_anthropic_client_async,
    get_anthropic_chat_completion,
    get_anthropic_chat_completion_async,
    parse_anthropic_completion
)

class DebateMethod:
    """
    Implements an AI debate method where two AI models engage in structured debate
    with multiple rounds of discussion.
    """
    
    def __init__(
        self,
        num_rounds: int = 3,
        max_tokens_per_response: int = 1000,
        temperature: float = 0.7,
        model_name: str = "claude-3-sonnet-20240229",
        judge_criteria: Optional[str] = None,
        async_mode: bool = True
    ):
        """Initialize debate method with configurable parameters."""
        self.num_rounds = num_rounds
        self.max_tokens = max_tokens_per_response
        self.temperature = temperature
        self.model_name = model_name
        self.judge_criteria = judge_criteria or self._default_judge_criteria()
        self.debate_history = []
        self.async_mode = async_mode
        self.topic = None  
        self.judgment_result = None 
        
        if async_mode:
            self.agent_a = get_anthropic_client_async()
            self.agent_b = get_anthropic_client_async()
            self.judge = get_anthropic_client_async()
        else:
            self.agent_a = get_anthropic_client_sync()
            self.agent_b = get_anthropic_client_sync()
            self.judge = get_anthropic_client_sync()
    
    def _default_judge_criteria(self) -> str:
        """Define default criteria for debate evaluation."""
        return """
        Please evaluate this debate carefully and provide:

        1. Numerical scores (1-10) for each criterion:
           - Evidence Quality: Rate the quality and reliability of evidence presented
           - Logical Consistency: Evaluate the coherence and validity of arguments
           - Counterargument Handling: Assess how well each side addressed opposing points
           - Clarity: Rate the clarity and precision of claims made

        2. A brief explanation for each score

        3. A final verdict determining:
           - Which side presented stronger arguments
           - Key deciding factors
           - Areas where each side excelled or could improve

        Format your response as:
        SCORES:
        Evidence Quality: [score]/10
        Logical Consistency: [score]/10
        Counterargument Handling: [score]/10
        Clarity: [score]/10

        EXPLANATIONS:
        [Provide brief explanations for each score]

        VERDICT: [name of winner agent and their stance]
        [Your final judgment and analysis]
        """
    
    def _format_debate_prompt(self, topic: str, stance: str) -> str:
        """Format the initial prompt for debaters."""
        return f"""
        You are participating in a formal debate about: {topic}
        Your stance: {stance}
        
        Guidelines:
        1. Make clear, factual claims that can be verified
        2. Support arguments with specific evidence
        3. Address counterarguments directly
        4. Maintain logical consistency
        5. Focus on the strongest available arguments
        
        Present your argument in a structured format:
        - Main claim
        - Supporting evidence (cite specific sources where possible)
        - Anticipated counterarguments and your responses
        - Conclusion reinforcing your main points
        """
    
    def _format_rebuttal_prompt(self, previous_argument: str) -> str:
        """Format prompt for rebuttals."""
        return f"""
        Respond to the following argument:
        {previous_argument}
        
        Guidelines:
        1. Address the strongest points directly
        2. Identify logical flaws or missing evidence
        3. Provide counter-evidence where applicable
        4. Present alternative interpretations of evidence
        5. Maintain focus on the core claims
        6. Strengthen your original position while engaging with opponent's points

        Structure your rebuttal clearly:
        - Direct responses to key points
        - New evidence supporting your position
        - Logical challenges to opponent's arguments
        - Reinforcement of your main thesis
        """

    async def _get_response_async(self, client: AsyncAnthropic, prompt: str) -> str:
        """Get async response from the model."""
        messages = [{"role": "user", "content": prompt}]
        response = await get_anthropic_chat_completion_async(
            client=client,
            messages=messages,
            model_name=self.model_name,
            max_new_tokens=self.max_tokens,
            temperature=self.temperature
        )
        return parse_anthropic_completion(response)
    
    def _get_response_sync(self, client: Anthropic, prompt: str) -> str:
        """Get sync response from the model."""
        messages = [{"role": "user", "content": prompt}]
        response = get_anthropic_chat_completion(
            client=client,
            messages=messages,
            model_name=self.model_name,
            max_new_tokens=self.max_tokens,
            temperature=self.temperature
        )
        return parse_anthropic_completion(response)

    async def conduct_debate(self, topic: str) -> Dict:
        """Conduct a full debate on the given topic."""
        self.topic = topic  # Store the topic
        
        # Initial args
        agent_a_prompt = self._format_debate_prompt(topic, "supporting")
        agent_b_prompt = self._format_debate_prompt(topic, "opposing")
        
        # First round
        if self.async_mode:
            response_a = await self._get_response_async(self.agent_a, agent_a_prompt)
            response_b = await self._get_response_async(self.agent_b, agent_b_prompt)
        else:
            response_a = self._get_response_sync(self.agent_a, agent_a_prompt)
            response_b = self._get_response_sync(self.agent_b, agent_b_prompt)
        
        self.debate_history.append({"round": 1, "agent": "A", "content": response_a})
        self.debate_history.append({"round": 1, "agent": "B", "content": response_b})
        
        # Subsequent rounds
        for round_num in range(2, self.num_rounds + 1):
            rebuttal_prompt_a = self._format_rebuttal_prompt(response_b)
            rebuttal_prompt_b = self._format_rebuttal_prompt(response_a)
            
            if self.async_mode:
                response_a = await self._get_response_async(self.agent_a, rebuttal_prompt_a)
                response_b = await self._get_response_async(self.agent_b, rebuttal_prompt_b)
            else:
                response_a = self._get_response_sync(self.agent_a, rebuttal_prompt_a)
                response_b = self._get_response_sync(self.agent_b, rebuttal_prompt_b)
            
            self.debate_history.append({"round": round_num, "agent": "A", "content": response_a})
            self.debate_history.append({"round": round_num, "agent": "B", "content": response_b})
        
        # Final judgment
        judgment = await self._get_judgment() if self.async_mode else self._get_judgment_sync()
        self.judgment_result = judgment 
        
        return {
            "topic": self.topic,
            "debate_history": self.debate_history,
            "judgment": judgment
        }

    def _parse_judgment_scores(self, judgment_text: str) -> Dict:
        """Parse numerical scores from judgment text using regex."""
        scores = {
            "agent": "zzz",
            "evidence_quality": 0,
            "logical_consistency": 0,
            "counterargument_handling": 0,
            "clarity": 0
        }
        
        try:
            patterns = {
                "agent": r"Verdict\s*(\d+)(?:/10)",
                "evidence_quality": r"Evidence Quality:\s*(\d+)(?:/10)?",
                "logical_consistency": r"Logical Consistency:\s*(\d+)(?:/10)?",
                "counterargument_handling": r"Counterargument Handling:\s*(\d+)(?:/10)?",
                "clarity": r"Clarity:\s*(\d+)(?:/10)?"
            }
            
            for key, pattern in patterns.items():
                match = re.search(pattern, judgment_text)
                if match:
                    scores[key] = int(match.group(1))
            
            return scores
        except Exception:
            return scores
        
    def _parse_verdict(self, judgment_text: str) -> Dict:
        """Parse verdict from judgment text using regex."""
        try:
            verdict_pattern = r"(?:VERDICT|Final Verdict):\s*([^.\n]*)"
            match = re.search(verdict_pattern, judgment_text, re.IGNORECASE)
            return {
                "verdict": match.group(1).strip() if match else "No verdict found"
            }
        except Exception:
            return {"verdict": "Error parsing verdict"}

    async def _get_judgment(self) -> Dict:
        """Get final judgment on the debate (async)."""
        debate_summary = self._format_debate_summary()
        judgment_prompt = self._format_judgment_prompt(debate_summary)
        judgment_response = await self._get_response_async(self.judge, judgment_prompt)
        
        return {
            "final_judgment": self._parse_verdict(judgment_response),
            "criteria_scores": self._parse_judgment_scores(judgment_response)
        }
    
    def _get_judgment_sync(self) -> Dict:
        """Get final judgment on the debate (sync)."""
        debate_summary = self._format_debate_summary()
        judgment_prompt = self._format_judgment_prompt(debate_summary)
        judgment_response = self._get_response_sync(self.judge, judgment_prompt)
        
        return {
            "final_judgment": judgment_response,
            "criteria_scores": self._parse_judgment_scores(judgment_response)
        }
    
    def _format_debate_summary(self) -> str:
        """Format the debate history for judgment."""
        return "\n\n".join([
            f"Round {entry['round']} - Agent {entry['agent']}:\n{entry['content']}"
            for entry in self.debate_history
        ])
    
    def _format_judgment_prompt(self, debate_summary: str) -> str:
        """Format the prompt for judgment."""
        return f"""
        You are serving as an impartial judge for the following debate on: {self.topic}
        
        Please evaluate carefully using these criteria:
        {self.judge_criteria}
        
        Debate history:
        {debate_summary}
        
        Provide your evaluation following the format specified in the criteria.
        """
    
    def save_debate_record(self, filename: str):
        """Save the complete debate record including history, criteria, and judgment."""
        debate_record = {
            "topic": self.topic,
            "debate_history": self.debate_history,
            "judge_criteria": self.judge_criteria,
            "judgment": {
                "result": self.judgment_result.get("final_judgment") if self.judgment_result else None,
                "scores": self.judgment_result.get("criteria_scores", {}) if self.judgment_result else {},
                "timestamp": datetime.datetime.now().isoformat()
            },
            "metadata": {
                "num_rounds": self.num_rounds,
                "model_name": self.model_name,
                "temperature": self.temperature
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(debate_record, f, indent=2)

    def load_debate_record(self, filename: str) -> Dict:
        """Load a saved debate record."""
        with open(filename, 'r') as f:
            return json.load(f)


async def main():
    # Example debate
    debate = DebateMethod(
        num_rounds=3,
        max_tokens_per_response=1000,
        temperature=0.7,
        async_mode=True,
        model_name="claude-3-sonnet-20240229"
    )
    
    # topic = "Is November a rainy season month in Costa Rica?"
    # topic = "Is authentic love possible in Sartre's method?"
    topic = "Was congestion pricing successful in London?"
    results = await debate.conduct_debate(topic)
    
    debate.save_debate_record("debate_record.json")
    
    loaded_record = debate.load_debate_record("debate_record.json")

    print(f"\nDebate Topic: {loaded_record['topic']}")
    print("\nFinal Judgment:", loaded_record["judgment"]["result"])
    print("\nScores:", json.dumps(loaded_record["judgment"]["scores"], indent=2))

if __name__ == "__main__":
    asyncio.run(main())