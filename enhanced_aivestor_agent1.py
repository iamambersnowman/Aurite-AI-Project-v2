# enhanced_aivestor_agent1.py
"""
Enhanced AIvestor Agent 1 - Adds Aurite agent capabilities

Author: Enhanced for Aurite AI Project
Version: 1.0.0
Description: Smart conversational AI for building comprehensive investor profiles
"""

import json
import logging
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, asdict

logger = logging.getLogger('AIvestor')

class EnhancedAIvestor:
    """
    Enhanced AIvestor - Adds smarter conversational capabilities to the original
    """
    
    def __init__(self):
        # Retain original profile structure
        self.profile = {
            'investment_amount': 10000,
            'risk_level': 'moderate',
            'time_horizon': 5,
            'investment_goal': None,
            'monthly_contribution': 0,
            'preferred_sectors': [],
            'avoid_sectors': [],
            'needs_liquidity': False,
            'prefers_esg': False,
            'tax_sensitive': False
        }
        self.context = {
            'turn_number': 0,
            'extracted_info': {},
            'confidence_scores': {}
        }
    
    def start_conversation(self) -> Dict[str, Any]:
        """
        Maintain original interface but enhance processing capabilities
        """
        print("\n" + "="*60)
        print("🤖 AIvestor - Your Personal Investment AI")
        print("="*60)
        
        # Turn 1 - Smarter opening
        response1 = input("\nWhat brings you to investing today? Tell me about your financial goals: ")
        self._process_turn_1(response1)
        
        # Turn 2 - Dynamically generate questions
        question2 = self._generate_adaptive_question()
        response2 = input(f"\n[Turn 2] {question2}")
        self._process_turn_2(response2)
        
        # Turn 3 - If needed
        if self._needs_more_info():
            question3 = self._generate_final_question()
            response3 = input(f"\n[Turn 3] {question3}")
            self._process_turn_3(response3)
        
        # Confirm and finalize
        self._finalize_profile()
        
        print("\n✅ Perfect! I've built your personalized investment profile.")
        print(f"   Profile ID: {self.profile['profile_id']}")
        
        return self.profile
    
    def _process_turn_1(self, response: str):
        """
        Process first turn - extract initial preferences
        """
        # Extract core information
        self._extract_amount(response)
        self._extract_timeline(response)
        self._extract_goals(response)
        self._analyze_sophistication(response)
    
    def _generate_adaptive_question(self) -> str:
        """
        Generate personalized questions based on known info
        """
        amount = self.profile.get('investment_amount', 10000)
        goal = self.profile.get('investment_goal')
        
        # Adjust question style based on amount
        if amount > 100000:
            return f"""With ${amount:,.0f} to invest, let's talk about risk.
How would you feel if your portfolio dropped 20% (${amount*0.2:,.0f}) temporarily?
   a) Great buying opportunity!
   b) Hold and wait for recovery
   c) Sell some to limit losses  
   d) Get out immediately"""
        
        elif goal == "retirement":
            return """For retirement planning, which approach fits you best?
   a) Maximum growth, I can handle volatility
   b) Balanced growth with moderate risk
   c) Steady growth, minimal volatility
   d) Preserve capital above all"""
        
        else:
            return """How would you describe your investment style?
   a) Aggressive - Maximum returns
   b) Moderate - Balanced approach
   c) Conservative - Safety first"""
    
    def _process_turn_2(self, response: str):
        """
        Enhanced Turn 2 processing - smarter risk assessment
        """
        response_lower = response.lower()
        
        # More detailed risk classification
        if 'a)' in response_lower or 'buying opportunity' in response_lower or 'aggressive' in response_lower:
            self.profile['risk_level'] = 'aggressive'
            self.profile['behavioral_traits'] = {'contrarian': True, 'patient': True}
        elif 'b)' in response_lower or 'hold' in response_lower or 'moderate' in response_lower or 'balanced' in response_lower:
            self.profile['risk_level'] = 'moderate'
            self.profile['behavioral_traits'] = {'disciplined': True}
        else:
            self.profile['risk_level'] = 'conservative'
            self.profile['behavioral_traits'] = {'risk_averse': True}
    
    def _needs_more_info(self) -> bool:
        """
        Intelligently determine if more information is needed
        """
        # Check critical information completeness
        required = ['investment_amount', 'risk_level', 'time_horizon']
        return not all(self.profile.get(field) for field in required)
    
    def _generate_final_question(self) -> str:
        """
        Generate final supplementary question
        """
        if not self.profile.get('preferred_sectors'):
            return "Any sectors you're particularly interested in? (Tech, Healthcare, Energy, etc)"
        elif self.profile.get('monthly_contribution') == 0:
            return "Will you be adding money regularly? (e.g., $500/month)"
        else:
            return "Anything else I should know? (ESG preferences, tax considerations, etc)"
    
    def _process_turn_3(self, response: str):
        """
        Process final supplementary information
        """
        # Extract sector preferences
        sectors_map = {
            'tech': 'Technology',
            'health': 'Healthcare',
            'finance': 'Financial',
            'energy': 'Energy',
            'retail': 'Consumer'
        }
        
        for keyword, sector in sectors_map.items():
            if keyword in response.lower():
                if 'avoid' in response.lower():
                    self.profile['avoid_sectors'].append(sector)
                else:
                    self.profile['preferred_sectors'].append(sector)
        
        # Extract monthly investment
        import re
        monthly_match = re.search(r'\$?(\d+(?:,\d+)*)\s*(?:monthly|/month|per month)', response.lower())
        if monthly_match:
            self.profile['monthly_contribution'] = float(monthly_match.group(1).replace(',', ''))
    
    def _finalize_profile(self):
        """
        Finalize profile and add metadata
        """
        # Add ID and timestamp
        self.profile['profile_id'] = f"AIV_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.profile['creation_timestamp'] = datetime.now().isoformat()
        
        # Calculate profile quality score
        info_completeness = sum(1 for v in self.profile.values() if v) / len(self.profile)
        self.profile['profile_quality'] = round(info_completeness, 2)
        
        # Add behavioral insights
        if not self.profile.get('behavioral_traits'):
            self.profile['behavioral_traits'] = self._infer_behavioral_traits()
        
        # Save profile to analysis_outputs for consistency with other agents
        self._save_profile()
    
    def _save_profile(self):
        """Save user profile to analysis_outputs folder"""
        # Ensure analysis_outputs directory exists
        output_dir = "analysis_outputs"
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"user_profile_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)
        
        try:
            with open(filepath, 'w') as f:
                json.dump(self.profile, f, indent=2)
            logger.info(f"✅ User profile saved to {filepath}")
            print(f"💾 Profile saved to: {filepath}")
        except Exception as e:
            logger.error(f"❌ Error saving profile: {e}")
    
    def _infer_behavioral_traits(self) -> Dict:
        """
        Infer user behavioral traits - demonstrates AI intelligence
        """
        traits = {}
        
        if self.profile['risk_level'] == 'aggressive':
            traits['risk_seeking'] = True
            traits['growth_focused'] = True
        elif self.profile['risk_level'] == 'conservative':
            traits['risk_averse'] = True
            traits['stability_focused'] = True
        
        if self.profile.get('monthly_contribution', 0) > 0:
            traits['disciplined_saver'] = True
        
        if self.profile.get('time_horizon', 0) > 10:
            traits['long_term_thinker'] = True
        
        return traits
    
    # Retain all original helper methods
    def _extract_amount(self, text: str):
        """Retain original amount extraction logic"""
        import re
        patterns = [
            r'\$?\s*(\d+(?:,\d+)*(?:\.\d+)?)\s*([kmb])?(?:illion)?',
            r'(\d+(?:,\d+)*)\s*dollars?',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                amount = float(match.group(1).replace(',', ''))
                unit = match.group(2) if len(match.groups()) > 1 else None
                
                if unit == 'k':
                    amount *= 1000
                elif unit == 'm':
                    amount *= 1000000
                
                self.profile['investment_amount'] = amount
                break
    
    def _extract_timeline(self, text: str):
        """Retain original timeline extraction logic"""
        import re
        patterns = [
            r'(\d+)\s*years?',
            r'next\s*(\d+)\s*years?',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                self.profile['time_horizon'] = int(match.group(1))
                break
        
        # Infer based on goals
        if 'retirement' in text.lower():
            self.profile['time_horizon'] = 20
        elif 'house' in text.lower():
            self.profile['time_horizon'] = 5
    
    def _extract_goals(self, text: str):
        """Retain original goal extraction logic"""
        goals_map = {
            'retirement': ['retire', 'retirement'],
            'house': ['house', 'home', 'property'],
            'education': ['college', 'education', 'university'],
            'wealth': ['grow', 'wealth', 'rich']
        }
        
        text_lower = text.lower()
        for goal, keywords in goals_map.items():
            if any(keyword in text_lower for keyword in keywords):
                self.profile['investment_goal'] = goal
                break
    
    def _analyze_sophistication(self, text: str):
        """Retain original user level analysis"""
        advanced_terms = ['etf', 'dividend', 'volatility', 'diversification']
        intermediate_terms = ['stocks', 'bonds', 'mutual fund']
        
        text_lower = text.lower()
        
        if any(term in text_lower for term in advanced_terms):
            self.profile['user_sophistication'] = 'advanced'
        elif any(term in text_lower for term in intermediate_terms):
            self.profile['user_sophistication'] = 'intermediate'
        else:
            self.profile['user_sophistication'] = 'beginner'

# Main function for workflow integration
def agent1_for_workflow() -> Dict[str, Any]:
    """
    Main entry point for Agent 1 in the workflow
    Returns user profile for portfolio construction
    """
    agent = EnhancedAIvestor()
    profile = agent.start_conversation()
    
    return profile

# Example usage
if __name__ == "__main__":
    # Run Agent 1
    user_profile = agent1_for_workflow()
    
    print("\n📊 Profile Summary:")
    print(f"   Amount: ${user_profile['investment_amount']:,.0f}")
    print(f"   Risk Level: {user_profile['risk_level']}")
    print(f"   Time Horizon: {user_profile['time_horizon']} years")
    print(f"   Goal: {user_profile.get('investment_goal', 'general wealth building')}")
    
    # The profile is now ready to be passed to Agent 4 (Portfolio Agent)
    print("\n✅ Profile ready for portfolio construction!")
