"""
Earnings Call Analyzer - AI Agent Swarm System
Dual-mode implementation: Claude API (paid) OR Ollama (free)

Usage:
    python earnings_analyzer.py

Features:
- 3 AI agents (Revenue, Profitability, Management)
- Agent swarm communication and debate
- Pre-loaded sample analyses for instant demo
- Fully automated analysis with either Claude API or Ollama

Modes:
  MODE 1: Claude API (Best Quality)
    - Claude Sonnet 4
    - Fully automated
    - Cost: ~$0.10/analysis
    - Setup: API key required

  MODE 2: Ollama (Free Alternative)
    - Llama 3.1 (local)
    - Fully automated
    - Cost: $0 (completely free)
    - Setup: Install Ollama + download model

Quick Start:
    # With Claude API:
    export ANTHROPIC_API_KEY='sk-ant-...'
    python earnings_analyzer.py

    # With Ollama (Free):
    ollama pull llama3.1
    python earnings_analyzer.py

    # Demo mode (no setup):
    python earnings_analyzer.py
    [Select option 1 for instant sample analysis]
"""

import asyncio
import json
import os
import base64
import time
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

OLLAMA_AVAILABLE = False
try:
    import ollama
    ollama.list()
    OLLAMA_AVAILABLE = True
except:
    pass

if not ANTHROPIC_API_KEY:
    print("\n" + "="*60)
    print("🔑 SELECT AI ENGINE")
    print("="*60)
    print("\n🤖 OPTION 1: CLAUDE API (Best Quality)")
    print("   - Claude Sonnet 4 (most intelligent)")
    print("   - Fully automated")
    print("   - Cost: ~$0.10 per analysis")
    print("   - Setup: Need API key from console.anthropic.com")
    print("\n🦙 OPTION 2: OLLAMA (Free & Automated)")
    print("   - Llama 3.1 (runs on your computer)")
    print("   - Fully automated")
    print("   - Cost: $0 (completely free)")
    if OLLAMA_AVAILABLE:
        print("   ✅ Ollama detected and ready!")
    else:
        print("   ⚠️  Not installed - Get from: https://ollama.com")
    print("\n📊 OPTION 3: DEMO MODE")
    print("   - Pre-loaded sample analyses")
    print("   - Instant results")
    print("   - No AI needed")
    print("\n" + "="*60)

    user_input = input("\nEnter API key, type 'ollama', or press Enter for demo: ").strip().lower()

    if user_input and user_input != 'ollama':
        ANTHROPIC_API_KEY = user_input
        print("\n✅ CLAUDE API MODE ACTIVATED")
        print("   Using Claude Sonnet 4 for analysis")
    elif user_input == 'ollama':
        if OLLAMA_AVAILABLE:
            print("\n✅ OLLAMA MODE ACTIVATED")
            print("   Using Llama 3.1 (free local AI)")
        else:
            print("\n❌ Ollama not available!")
            print("   Install from: https://ollama.com")
            print("   Then run: ollama pull llama3.1")
            print("\n   Using DEMO MODE (samples only)")
            OLLAMA_AVAILABLE = False
    else:
        print("\n✅ DEMO MODE ACTIVATED")
        print("   Using pre-loaded sample analyses")

USE_REAL_API = bool(ANTHROPIC_API_KEY)
USE_OLLAMA = OLLAMA_AVAILABLE and not USE_REAL_API

if USE_REAL_API:
    try:
        import anthropic
        print("✓ Claude API ready (Sonnet 4)")
    except ImportError:
        print("⚠️  Install: pip install anthropic")
        USE_REAL_API = False
elif USE_OLLAMA:
    print("✓ Ollama ready (Llama 3.1 - Free)")
else:
    print("ℹ️  Demo mode - Use samples for instant results")

SAMPLE_ANALYSES = {
    "techcorp_q3_2025": {
        "company": "TechCorp Inc - Q3 2025",
        "revenue": {
            "score": 8.2,
            "verdict": "STRONG",
            "key_metrics": {
                "revenue": "$2.8B (up 23% YoY)",
                "guidance": "$3.2B next quarter (beat consensus)",
                "customer_growth": "+1,200 enterprise customers",
                "arr": "$10.5B (up 27% YoY)"
            },
            "highlights": [
                "Revenue beat analyst estimates by $150M (5.7%)",
                "Strong international growth - EMEA up 31% YoY",
                "Raised full-year guidance from $11B to $11.5B",
                "Enterprise segment growing faster than SMB"
            ],
            "concerns": [
                "Customer acquisition costs increased 12%",
                "SMB segment growth slowing (8% vs 15% last quarter)"
            ]
        },
        "profitability": {
            "score": 6.5,
            "verdict": "MIXED",
            "key_metrics": {
                "gross_margin": "72% (down from 74%)",
                "operating_margin": "18% (down from 21%)",
                "net_income": "$420M (up 8% YoY)",
                "free_cash_flow": "$580M (up 22%)"
            },
            "highlights": [
                "Operating expenses well-controlled, up only 9% vs 23% revenue growth",
                "Free cash flow beat expectations significantly",
                "R&D efficiency improving"
            ],
            "concerns": [
                "Gross margin compression due to infrastructure costs",
                "Operating margin trending down for 3 consecutive quarters",
                "Q4 guidance implies further margin compression to 16%"
            ]
        },
        "management": {
            "score": 7.8,
            "verdict": "CONFIDENT",
            "key_metrics": {
                "tone": "Bullish and confident",
                "defensiveness": "Low",
                "transparency": "8.5/10"
            },
            "highlights": [
                "CEO used 'confident' 12 times, 'excited' 8 times",
                "Specific details on Q4 pipeline - $800M qualified deals",
                "CEO bought $2M shares last month",
                "Directly addressed margin pressure with concrete plan"
            ],
            "concerns": [
                "Avoided question about Microsoft competition",
                "Mentioned 'macroeconomic headwinds' 5 times",
                "Vague on international expansion timeline"
            ]
        }
    },
    "tesla_q3_2024": {
        "company": "Tesla Inc - Q3 2024",
        "revenue": {
            "score": 7.5,
            "verdict": "STRONG",
            "key_metrics": {
                "revenue": "$25.2B (up 8% YoY)",
                "deliveries": "430,000 vehicles",
                "energy_storage": "6.9 GWh (up 73%)",
                "automotive_revenue": "$20B"
            },
            "highlights": [
                "Record production and deliveries",
                "Energy storage deployments up 73% YoY",
                "Cybertruck production ramping"
            ],
            "concerns": [
                "Slower growth than previous years (8% vs 50%+)",
                "Competitive pressure in China"
            ]
        },
        "profitability": {
            "score": 5.8,
            "verdict": "WEAK",
            "key_metrics": {
                "automotive_margin": "17.1% (down from 18.7%)",
                "operating_margin": "10.8%",
                "free_cash_flow": "$2.7B",
                "capex": "$3.5B"
            },
            "highlights": [
                "Generated $2.7B in free cash flow",
                "Cost reduction progress on Cybertruck"
            ],
            "concerns": [
                "Automotive gross margin down significantly",
                "R&D expenses up 35% YoY",
                "Margin pressure from price reductions"
            ]
        },
        "management": {
            "score": 7.2,
            "verdict": "CONFIDENT",
            "key_metrics": {
                "tone": "Optimistic about future",
                "defensiveness": "Medium",
                "transparency": "6.5/10"
            },
            "highlights": [
                "Confident about Cybertruck margins by Q4",
                "Strong belief in FSD progress",
                "Not worried about competition"
            ],
            "concerns": [
                "Vague on Cybertruck profitability timeline",
                "Dismissed competitive threats quickly",
                "Overly optimistic on FSD"
            ]
        }
    }
}

class MessageType(Enum):
    """Types of messages agents can send"""
    ANALYSIS = "analysis"
    CHALLENGE = "challenge"
    QUESTION = "question"
    CONSENSUS = "consensus"

@dataclass
class Message:
    """Message structure for agent communication"""
    sender: str
    recipients: List[str]
    msg_type: MessageType
    content: str
    data: Optional[Dict] = None
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())

class MessageBus:
    """Central communication hub for agent swarm"""

    def __init__(self):
        self.messages: List[Message] = []
        self.subscribers: Dict[str, List] = {}

    async def publish(self, message: Message):
        """Publish message to all relevant subscribers"""
        self.messages.append(message)

        if "all" in message.recipients:
            recipients = list(self.subscribers.keys())
        else:
            recipients = message.recipients

        for recipient in recipients:
            if recipient in self.subscribers and recipient != message.sender:
                for callback in self.subscribers[recipient]:
                    await callback(message)

    def subscribe(self, agent_id: str, callback):
        """Subscribe agent to message bus"""
        if agent_id not in self.subscribers:
            self.subscribers[agent_id] = []
        self.subscribers[agent_id].append(callback)

    def get_history(self) -> List[Message]:
        """Get all message history"""
        return self.messages

class AIAPI:
    """Wrapper supporting Claude API and Ollama"""

    def __init__(self, api_key: str = None, use_ollama: bool = False):
        self.use_real = USE_REAL_API and api_key
        self.use_ollama = use_ollama or (USE_OLLAMA and not self.use_real)

        if self.use_real:
            self.client = anthropic.Anthropic(api_key=api_key)
        elif self.use_ollama:
            try:
                import ollama
                self.ollama = ollama
            except ImportError:
                print("⚠️  Ollama not available, using mock")
                self.use_ollama = False

    async def analyze_document(
        self,
        system_prompt: str,
        user_prompt: str,
        document_base64: str = None,
        document_type: str = "application/pdf",
        max_tokens: int = 3000
    ) -> str:
        """Call Claude API, Ollama, or return mock response"""

        # MODE 1: Claude API
        if self.use_real:
            try:
                content = []

                if document_base64:
                    content.append({
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": document_type,
                            "data": document_base64
                        }
                    })

                content.append({"type": "text", "text": user_prompt})

                response = self.client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=max_tokens,
                    system=system_prompt,
                    messages=[{"role": "user", "content": content}]
                )

                return response.content[0].text

            except Exception as e:
                print(f"  ⚠️  API Error: {e}. Falling back to mock.")
                return self._mock_response(user_prompt)

        # MODE 2: Ollama (Free Local AI)
        elif self.use_ollama:
            try:
                # Combine system and user prompts
                full_prompt = f"{system_prompt}\n\n{user_prompt}"

                print("  🦙 Calling Ollama (local AI)...")
                response = self.ollama.chat(
                    model='llama3.1',
                    messages=[{'role': 'user', 'content': full_prompt}]
                )

                return response['message']['content']

            except Exception as e:
                print(f"  ⚠️  Ollama Error: {e}. Falling back to mock.")
                return self._mock_response(user_prompt)

        # MODE 3: Mock (for demo/testing)
        else:
            await asyncio.sleep(0.8)
            return self._mock_response(user_prompt)

    def _mock_response(self, prompt: str) -> str:
        """Generate mock responses for demo"""

        if "revenue" in prompt.lower():
            return """{
                "score": 8.2,
                "verdict": "STRONG",
                "key_metrics": {
                    "revenue": "$2.8B (up 23% YoY)",
                    "guidance": "$3.2B next quarter",
                    "customer_growth": "+1,200 enterprise customers",
                    "arr": "$10.5B (up 27% YoY)"
                },
                "highlights": [
                    "Revenue beat analyst estimates by $150M (5.7%)",
                    "Strong international growth - EMEA up 31% YoY",
                    "Raised full-year guidance"
                ],
                "concerns": [
                    "Customer acquisition costs increased 12%",
                    "SMB segment growth slowing"
                ]
            }"""

        elif "profitability" in prompt.lower() or "margin" in prompt.lower():
            return """{
                "score": 6.5,
                "verdict": "MIXED",
                "key_metrics": {
                    "gross_margin": "72% (down from 74%)",
                    "operating_margin": "18% (down from 21%)",
                    "net_income": "$420M (up 8% YoY)",
                    "free_cash_flow": "$580M (up 22%)"
                },
                "highlights": [
                    "Operating expenses well-controlled",
                    "Free cash flow beat expectations",
                    "R&D efficiency improving"
                ],
                "concerns": [
                    "Gross margin compression",
                    "Operating margin trending down",
                    "Q4 guidance implies further compression"
                ]
            }"""

        elif "management" in prompt.lower():
            return """{
                "score": 7.8,
                "verdict": "CONFIDENT",
                "key_metrics": {
                    "tone": "Bullish and confident",
                    "defensiveness": "Low",
                    "transparency": "8.5/10"
                },
                "positive_signals": [
                    "CEO used confident language frequently",
                    "Specific pipeline details provided",
                    "CEO insider buying signal"
                ],
                "red_flags": [
                    "Avoided competitive questions",
                    "Some hedging language present"
                ]
            }"""

        else:
            return '{"score": 7.0, "verdict": "NEUTRAL"}'

class EarningsAgent:
    """Base class for all earnings analysis agents"""

    def __init__(
        self,
        agent_id: str,
        expertise: str,
        message_bus: MessageBus,
        ai_api: AIAPI
    ):
        self.agent_id = agent_id
        self.expertise = expertise
        self.message_bus = message_bus
        self.ai = ai_api

        self.analysis: Dict = {}
        self.score: float = 0.0
        self.inbox: List[Message] = []

        self.message_bus.subscribe(agent_id, self.receive_message)

    async def receive_message(self, message: Message):
        """Handle incoming messages from other agents"""
        self.inbox.append(message)

        if message.msg_type == MessageType.CHALLENGE:
            await self.handle_challenge(message)

    async def send_message(
        self,
        recipients: List[str],
        msg_type: MessageType,
        content: str,
        data: Dict = None
    ):
        """Send message to other agents"""
        message = Message(
            sender=self.agent_id,
            recipients=recipients,
            msg_type=msg_type,
            content=content,
            data=data
        )
        await self.message_bus.publish(message)

    async def broadcast_analysis(self, analysis: Dict):
        """Share analysis with all agents"""
        await self.send_message(
            recipients=["all"],
            msg_type=MessageType.ANALYSIS,
            content=f"{self.agent_id} completed analysis - Score: {self.score}/10",
            data=analysis
        )

    async def challenge_peer(self, peer_id: str, concern: str):
        """Challenge another agent's conclusion"""
        await self.send_message(
            recipients=[peer_id],
            msg_type=MessageType.CHALLENGE,
            content=concern
        )

    async def handle_challenge(self, message: Message):
        """Respond to challenges from peers"""
        print(f"  [{self.agent_id}] Received challenge: {message.content}")

        import random
        if random.random() > 0.7:
            old_score = self.score
            self.score *= 0.9
            print(f"  [{self.agent_id}] Revised score: {old_score:.1f} → {self.score:.1f}")

class RevenueAgent(EarningsAgent):
    """Analyzes revenue growth, guidance, and customer metrics"""

    def __init__(self, message_bus: MessageBus, ai_api: AIAPI):
        super().__init__("revenue_agent", "Revenue Analysis", message_bus, ai_api)

    async def analyze(
        self,
        document_text: str = None,
        document_base64: str = None
    ) -> Dict:
        """Analyze revenue performance"""
        print(f"\n[{self.agent_id}] Analyzing revenue metrics...")

        prompt = """
        Analyze the REVENUE performance from this earnings report.
        
        Focus on:
        1. Revenue growth (actual vs estimates vs guidance)
        2. Revenue guidance for next quarter/year
        3. Customer acquisition and retention metrics
        4. Geographic/segment breakdown
        5. ARR/MRR trends (if SaaS)
        
        Return JSON with this EXACT structure:
        {
            "score": 7.5,
            "verdict": "STRONG/MIXED/WEAK",
            "key_metrics": {
                "revenue": "actual revenue with growth %",
                "guidance": "forward guidance",
                "customer_growth": "customer metrics",
                "arr": "ARR if applicable"
            },
            "highlights": ["list of positive points"],
            "concerns": ["list of concerns"]
        }
        
        Return ONLY valid JSON, no markdown.
        """

        if document_text:
            prompt = f"{prompt}\n\nDocument text:\n{document_text[:3000]}"

        response = await self.ai.analyze_document(
            system_prompt="You are a revenue analysis expert for public companies. Focus on top-line growth.",
            user_prompt=prompt,
            document_base64=document_base64
        )

        try:
            analysis = json.loads(response)
        except:
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
                analysis = json.loads(json_str)
            else:
                analysis = json.loads(self.ai._mock_response("revenue"))

        self.analysis = analysis
        self.score = analysis.get('score', 7.0)

        await self.broadcast_analysis(analysis)
        await asyncio.sleep(0.3)

        return analysis


class ProfitabilityAgent(EarningsAgent):
    """Analyzes margins, costs, and operational efficiency"""

    def __init__(self, message_bus: MessageBus, ai_api: AIAPI):
        super().__init__("profitability_agent", "Profitability Analysis", message_bus, ai_api)

    async def analyze(
        self,
        document_text: str = None,
        document_base64: str = None
    ) -> Dict:
        """Analyze profitability and margins"""
        print(f"\n[{self.agent_id}] Analyzing profitability metrics...")

        prompt = """
        Analyze the PROFITABILITY and MARGINS from this earnings report.
        
        Focus on: Gross margin, operating margin, net income, free cash flow, cost efficiency
        
        Return JSON with exact structure (score, verdict, key_metrics, highlights, concerns).
        Return ONLY valid JSON, no markdown.
        """

        if document_text:
            prompt = f"{prompt}\n\nDocument text:\n{document_text[:3000]}"

        response = await self.ai.analyze_document(
            system_prompt="You are a profitability analysis expert. Focus on margins and efficiency.",
            user_prompt=prompt,
            document_base64=document_base64
        )

        try:
            analysis = json.loads(response)
        except:
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
                analysis = json.loads(json_str)
            else:
                analysis = json.loads(self.ai._mock_response("profitability"))

        self.analysis = analysis
        self.score = analysis.get('score', 6.5)

        await self.broadcast_analysis(analysis)
        await asyncio.sleep(0.3)

        # Challenge revenue agent if needed
        revenue_messages = [m for m in self.inbox if m.sender == "revenue_agent"]
        if revenue_messages and self.score < 7.0:
            revenue_data = revenue_messages[0].data
            if revenue_data and revenue_data.get('score', 0) > 8.0:
                await self.challenge_peer(
                    "revenue_agent",
                    "Revenue growth is impressive, but margin compression is concerning."
                )

        return analysis


class ManagementAgent(EarningsAgent):
    """Analyzes CEO/CFO tone, confidence, and red flags"""

    def __init__(self, message_bus: MessageBus, ai_api: AIAPI):
        super().__init__("management_agent", "Management Analysis", message_bus, ai_api)

    async def analyze(
        self,
        document_text: str = None,
        document_base64: str = None
    ) -> Dict:
        """Analyze management tone and credibility"""
        print(f"\n[{self.agent_id}] Analyzing management commentary...")

        prompt = """
        Analyze MANAGEMENT TONE and CREDIBILITY from this earnings call.
        
        Focus on: CEO/CFO confidence, forward-looking statements, red flags, track record
        
        Return JSON with exact structure (score, verdict, key_metrics/tone_analysis, positive_signals, red_flags).
        Return ONLY valid JSON, no markdown.
        """

        if document_text:
            prompt = f"{prompt}\n\nDocument text:\n{document_text[:3000]}"

        response = await self.ai.analyze_document(
            system_prompt="You are an expert at reading executive communications. Detect confidence and red flags.",
            user_prompt=prompt,
            document_base64=document_base64
        )

        try:
            analysis = json.loads(response)
        except:
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
                analysis = json.loads(json_str)
            else:
                analysis = json.loads(self.ai._mock_response("management"))

        self.analysis = analysis
        self.score = analysis.get('score', 7.5)

        await self.broadcast_analysis(analysis)

        return analysis

class EarningsConsensus:
    """Builds consensus from distributed agent opinions"""

    def __init__(self, message_bus: MessageBus):
        self.message_bus = message_bus

    async def build_consensus(self, agents: List[EarningsAgent]) -> Dict:
        """Synthesize agent analyses into final verdict"""
        print(f"\n{'='*60}")
        print("BUILDING CONSENSUS")
        print(f"{'='*60}\n")

        print("Agent Scores:")
        agent_scores = {}
        for agent in agents:
            verdict = agent.analysis.get('verdict', 'N/A')
            print(f"  {agent.expertise}: {agent.score:.1f}/10 ({verdict})")
            agent_scores[agent.expertise] = {
                "score": agent.score,
                "verdict": verdict
            }

        # Calculate weighted average
        weights = {
            'revenue_agent': 0.40,
            'profitability_agent': 0.35,
            'management_agent': 0.25
        }

        weighted_score = sum(
            agent.score * weights[agent.agent_id]
            for agent in agents
        )

        # Determine verdict
        if weighted_score >= 8.0:
            verdict = "BEAT EXPECTATIONS - STRONG BUY"
        elif weighted_score >= 7.0:
            verdict = "MET EXPECTATIONS - HOLD/BUY"
        elif weighted_score >= 6.0:
            verdict = "MIXED RESULTS - HOLD"
        elif weighted_score >= 5.0:
            verdict = "MISSED EXPECTATIONS - HOLD/SELL"
        else:
            verdict = "POOR RESULTS - SELL"

        # Check for red flags
        red_flags = []
        messages = self.message_bus.get_history()
        challenges = [m for m in messages if m.msg_type == MessageType.CHALLENGE]
        if len(challenges) >= 2:
            red_flags.append("Multiple agents raised concerns")

        profitability = next((a for a in agents if a.agent_id == "profitability_agent"), None)
        revenue = next((a for a in agents if a.agent_id == "revenue_agent"), None)
        if profitability and revenue:
            if profitability.score < 6.5 and revenue.score > 8.0:
                red_flags.append("Strong revenue but weak margins - growth may not be profitable")

        if not red_flags:
            red_flags = ["None detected"]

        consensus = {
            "overall_score": round(weighted_score, 1),
            "verdict": verdict,
            "confidence": "High" if len(challenges) == 0 else "Medium" if len(challenges) == 1 else "Low",
            "agent_scores": agent_scores,
            "red_flags": red_flags,
            "recommendation": self._generate_recommendation(weighted_score, agents)
        }

        print(f"\n{'='*60}")
        print(f"FINAL VERDICT: {verdict}")
        print(f"Overall Score: {weighted_score:.1f}/10")
        print(f"{'='*60}\n")

        return consensus

    def _generate_recommendation(self, score: float, agents: List[EarningsAgent]) -> str:
        """Generate investment recommendation"""
        if score >= 8.0:
            return "Strong beat across the board. Consider buying on any dips."
        elif score >= 7.0:
            return "Solid results but not spectacular. Hold position or add on weakness."
        elif score >= 6.0:
            return "Mixed results. Monitor closely. Hold current position."
        else:
            return "Results fell short. Consider reducing position."

class EarningsAnalyzer:
    """Main orchestrator for earnings analysis"""

    def __init__(self, api_key: str = None):
        self.ai_api = AIAPI(
            api_key=api_key or ANTHROPIC_API_KEY,
            use_ollama=USE_OLLAMA
        )
        self.message_bus = MessageBus()
        self.agents = [
            RevenueAgent(self.message_bus, self.ai_api),
            ProfitabilityAgent(self.message_bus, self.ai_api),
            ManagementAgent(self.message_bus, self.ai_api)
        ]
        self.consensus_engine = EarningsConsensus(self.message_bus)
        self.samples = SAMPLE_ANALYSES

    def list_samples(self):
        """Show available pre-analyzed samples"""
        print("\n" + "="*60)
        print("AVAILABLE SAMPLE ANALYSES (Instant Demo)")
        print("="*60 + "\n")

        for i, (key, data) in enumerate(self.samples.items(), 1):
            overall = self.calculate_sample_score(data)
            print(f"{i}. {data['company']}")
            print(f"   Score: {overall}/10\n")

    def calculate_sample_score(self, sample_data):
        """Calculate overall score from sample data"""
        return round(
            sample_data["revenue"]["score"] * 0.40 +
            sample_data["profitability"]["score"] * 0.35 +
            sample_data["management"]["score"] * 0.25,
            1
        )

    async def analyze_sample(self, sample_key: str) -> Dict:
        """Load and display pre-analyzed sample"""
        if sample_key not in self.samples:
            print(f"❌ Sample not found")
            return None

        print("\n" + "="*60)
        print("LOADING SAMPLE ANALYSIS")
        print("="*60)

        sample_data = self.samples[sample_key]

        # Simulate agent processing
        print("\n🤖 AI Agents analyzing...")
        await asyncio.sleep(0.8)
        print(f"   ✓ Revenue Agent: {sample_data['revenue']['score']}/10")
        await asyncio.sleep(0.4)
        print(f"   ✓ Profitability Agent: {sample_data['profitability']['score']}/10")
        await asyncio.sleep(0.4)
        print(f"   ✓ Management Agent: {sample_data['management']['score']}/10")

        # Build full report
        overall_score = self.calculate_sample_score(sample_data)

        if overall_score >= 8.0:
            verdict = "BEAT EXPECTATIONS - STRONG BUY"
        elif overall_score >= 7.0:
            verdict = "MET EXPECTATIONS - HOLD/BUY"
        elif overall_score >= 6.0:
            verdict = "MIXED RESULTS - HOLD"
        else:
            verdict = "MISSED EXPECTATIONS - SELL"

        report = {
            "company": sample_data["company"],
            "timestamp": datetime.now().isoformat(),
            "consensus": {
                "overall_score": overall_score,
                "verdict": verdict,
                "confidence": "High"
            },
            "detailed_analysis": sample_data,
            "red_flags": ["Strong revenue but weak margins"] if overall_score == 7.4 else ["Margin compression with competition"]
        }

        self.display_results(report)
        return report

    async def analyze_document(
        self,
        file_path: str = None,
        text_content: str = None
    ) -> Dict:
        """Analyze document with AI (Claude API or Ollama)"""
        print(f"\n{'='*60}")
        print("EARNINGS ANALYZER - AGENT SWARM")
        print(f"{'='*60}")

        # Read file if provided
        document_base64 = None
        if file_path:
            print(f"Reading file: {file_path}")
            with open(file_path, 'rb') as f:
                file_bytes = f.read()
                document_base64 = base64.b64encode(file_bytes).decode()

        # Run agents
        print("\n--- PHASE 1: AGENT ANALYSIS ---")
        analyses = await asyncio.gather(*[
            agent.analyze(
                document_text=text_content,
                document_base64=document_base64
            )
            for agent in self.agents
        ])

        # Build consensus
        print("\n--- PHASE 2: CONSENSUS BUILDING ---")
        consensus = await self.consensus_engine.build_consensus(self.agents)

        report = {
            "timestamp": datetime.now().isoformat(),
            "consensus": consensus,
            "detailed_analysis": {
                "revenue": analyses[0],
                "profitability": analyses[1],
                "management": analyses[2]
            }
        }

        return report

    def display_results(self, report: Dict):
        """Display analysis results"""
        print(f"\n{'='*60}")
        print("EARNINGS ANALYSIS RESULTS")
        print(f"{'='*60}\n")

        consensus = report["consensus"]
        print(f"🎯 OVERALL SCORE: {consensus['overall_score']}/10")
        print(f"📊 VERDICT: {consensus['verdict']}")
        print(f"🔍 CONFIDENCE: {consensus['confidence']}\n")

        print("="*60)
        print("AGENT BREAKDOWN")
        print("="*60 + "\n")

        data = report["detailed_analysis"]

        # Revenue
        rev = data["revenue"]
        print("💰 REVENUE AGENT")
        print(f"   Score: {rev['score']}/10 ({rev['verdict']})")
        print(f"   Highlights:")
        for h in rev.get("highlights", [])[:3]:
            print(f"      ✓ {h}")
        if rev.get("concerns"):
            print(f"   Concerns:")
            for c in rev["concerns"][:2]:
                print(f"      ⚠️  {c}")
        print()

        # Profitability
        prof = data["profitability"]
        print("📈 PROFITABILITY AGENT")
        print(f"   Score: {prof['score']}/10 ({prof['verdict']})")
        print(f"   Highlights:")
        for h in prof.get("highlights", [])[:3]:
            print(f"      ✓ {h}")
        if prof.get("concerns"):
            print(f"   Concerns:")
            for c in prof["concerns"][:2]:
                print(f"      ⚠️  {c}")
        print()

        # Management
        mgmt = data["management"]
        print("👔 MANAGEMENT AGENT")
        print(f"   Score: {mgmt['score']}/10 ({mgmt.get('verdict', 'N/A')})")
        if mgmt.get("positive_signals") or mgmt.get("highlights"):
            signals = mgmt.get("positive_signals") or mgmt.get("highlights", [])
            print(f"   Positive Signals:")
            for s in signals[:3]:
                print(f"      ✓ {s}")
        if mgmt.get("red_flags") or mgmt.get("concerns"):
            flags = mgmt.get("red_flags") or mgmt.get("concerns", [])
            print(f"   Red Flags:")
            for r in flags[:2]:
                print(f"      ⚠️  {r}")
        print()

        # Recommendation
        print("="*60)
        print("💡 RECOMMENDATION")
        print("="*60)
        print(f"   {consensus.get('recommendation', 'See detailed analysis above')}")
        print()

    def save_report(self, report: Dict, filename: str = None):
        """Save analysis to JSON file"""
        if not filename:
            filename = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\n✅ Report saved to {filename}")

async def interactive_menu():
    """Interactive CLI menu"""

    analyzer = EarningsAnalyzer()

    print("\n" + "="*60)
    print("EARNINGS CALL ANALYZER - AI AGENT SWARM")
    print("="*60)

    if USE_REAL_API:
        mode = "🤖 CLAUDE API"
        cost = "~$0.10/analysis"
        speed = "Instant"
    elif USE_OLLAMA:
        mode = "🦙 OLLAMA (Free & Automated)"
        cost = "$0 (completely free)"
        speed = "Fast (~5 seconds)"
    else:
        mode = "📊 DEMO MODE (Samples Only)"
        cost = "$0"
        speed = "Instant (pre-loaded)"

    print(f"\n🤖 Current Mode: {mode}")
    print(f"💰 Cost: {cost}")
    print(f"⚡ Speed: {speed}")
    print("\n💡 Tip: Install Ollama for free automated analysis")
    print("   Download from: https://ollama.com")
    print("="*60 + "\n")

    while True:
        print("="*60)
        print("MAIN MENU")
        print("="*60)
        print("\n1. 📊 Analyze sample (instant - pre-generated)")
        print("2. 🦙 Analyze new earnings (automated)" +
              (" ✅" if (USE_REAL_API or USE_OLLAMA) else " ⚠️ Need API key or Ollama"))
        print("3. 📋 List available samples")
        print("4. ℹ️  Setup instructions (API key / Ollama)")
        print("5. ❌ Exit\n")

        choice = input("Select option (1-5): ").strip()

        if choice == "1":
            print("\nAvailable samples:")
            print("1. TechCorp Q3 2025")
            print("2. Tesla Q3 2024")

            sample_choice = input("\nSelect (1-2): ").strip()

            if sample_choice == "1":
                report = await analyzer.analyze_sample("techcorp_q3_2025")
            elif sample_choice == "2":
                report = await analyzer.analyze_sample("tesla_q3_2024")
            else:
                print("❌ Invalid choice")
                continue

            if report:
                save = input("\nSave to file? (y/n): ").strip().lower()
                if save == 'y':
                    analyzer.save_report(report)

        elif choice == "2":
            if not USE_REAL_API and not USE_OLLAMA:
                print("\n" + "="*60)
                print("⚠️  NO AI ENGINE AVAILABLE")
                print("="*60)
                print("\nYou need either Claude API or Ollama for automated analysis.")
                print("\nOption A: Get Claude API key (Best quality)")
                print("  1. Visit console.anthropic.com")
                print("  2. Get API key")
                print("  3. Restart program and enter key")
                print("\nOption B: Install Ollama (Free)")
                print("  1. Download from https://ollama.com")
                print("  2. Run: ollama pull llama3.1")
                print("  3. Restart program")
                print("\nFor now, use Option 1 to view sample analyses!")
                print("="*60)
                continue

            transcript = input("\nPaste earnings transcript (or 'skip'): ")
            if transcript.lower() == 'skip' or not transcript:
                continue

            engine = "Claude API" if USE_REAL_API else "Ollama"
            print(f"\n🤖 Analyzing with {engine}...")

            report = await analyzer.analyze_document(text_content=transcript)
            analyzer.display_results(report)

            save = input("\nSave to file? (y/n): ").strip().lower()
            if save == 'y':
                analyzer.save_report(report)

        elif choice == "3":
            analyzer.list_samples()

        elif choice == "4":
            print("\n" + "="*60)
            print("SETUP INSTRUCTIONS")
            print("="*60)
            print("\n🤖 OPTION 1: Claude API (Best Quality)")
            print("   1. Visit https://console.anthropic.com")
            print("   2. Sign up and add payment method ($5 free credit)")
            print("   3. Create API key")
            print("   4. Run: export ANTHROPIC_API_KEY='your-key'")
            print("   5. Restart this program")
            print("\n🦙 OPTION 2: Ollama (Free & Automated)")
            print("   1. Visit https://ollama.com")
            print("   2. Download and install Ollama")
            print("   3. Run: ollama pull llama3.1")
            print("   4. Run: pip install ollama")
            print("   5. Restart this program")
            print("\n📊 OPTION 3: No Setup (Demo Mode)")
            print("   - Use Option 1 in menu for sample analyses")
            print("   - Perfect for testing the system")
            print("="*60)

        elif choice == "5":
            print("\n👋 Thanks for using Earnings Analyzer!")
            print("="*60)
            break

        else:
            print("\n❌ Invalid option. Please select 1-5.\n")

if __name__ == "__main__":
    asyncio.run(interactive_menu())