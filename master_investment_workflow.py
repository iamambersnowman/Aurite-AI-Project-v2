###Original version that has to run stock analysis in the process
# master_investment_workflow.py
"""
Master Investment Workflow - Aurite AI Project
Orchestrates Agent 1 → Agent 2 → Agent 3 for complete investment advisory

Author: Aurite AI Team
Version: 1.0.0
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Dict, Any, List
from pathlib import Path
from dataclasses import asdict

# Import our agents
from enhanced_aivestor_agent1 import agent1_for_workflow
from stock_analysis_agent import StockAnalysisAgent, AgentConfig, LLMConfig
from gold_analysis_agent import GoldAnalysisAgent
from etf_analysis_agent import BondAnalysisAgent
from enhanced_macro_analysis import EnhancedMacroAnalyzer
from portfolio_agent import ProfessionalPortfolioOptimizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('MasterWorkflow')

class AuriteInvestmentWorkflow:
    """
    Master workflow orchestrator for the complete investment advisory system
    """
    
    def __init__(self):
        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.results = {
            'session_id': self.session_id,
            'timestamp': datetime.now().isoformat(),
            'agent1_profile': None,
            'agent2_analysis': {},
            'agent3_portfolio': None,
            'final_recommendation': None
        }
        
        # Ensure output directory exists
        os.makedirs("analysis_outputs", exist_ok=True)
        
        # Initialize portfolio agent
        self.portfolio_agent = ProfessionalPortfolioOptimizer()
        
    async def run_complete_workflow(self) -> Dict[str, Any]:
        """
        Execute the complete 3-agent investment workflow
        """
        print("🚀 AURITE AI INVESTMENT ADVISOR")
        print("=" * 60)
        print("Complete Investment Analysis & Portfolio Construction")
        print("=" * 60)
        
        try:
            # AGENT 1: User Profiling
            print("\n🤖 STEP 1: User Preference Analysis")
            print("-" * 40)
            user_profile = self._run_agent1()
            self.results['agent1_profile'] = user_profile
            
            # AGENT 2: Market & Asset Analysis  
            print("\n📊 STEP 2: Market & Asset Analysis")
            print("-" * 40)
            analysis_results = await self._run_agent2(user_profile)
            self.results['agent2_analysis'] = analysis_results
            
            # AGENT 3: Portfolio Construction
            print("\n🎯 STEP 3: Portfolio Construction")
            print("-" * 40)
            portfolio = await self._run_agent3(user_profile, analysis_results)
            self.results['agent3_portfolio'] = portfolio
            
            # Generate Final Recommendation
            final_recommendation = self._generate_final_output()
            self.results['final_recommendation'] = final_recommendation
            
            # Save complete results
            self._save_complete_results()
            
            print("\n✅ WORKFLOW COMPLETE!")
            print("=" * 60)
            
            return self.results
            
        except Exception as e:
            logger.error(f"Workflow failed: {e}")
            return {"error": str(e), "session_id": self.session_id}
    
    def _run_agent1(self) -> Dict[str, Any]:
        """
        Run Agent 1: User Preference Collection
        """
        print("Collecting your investment preferences...")
        
        # Run the enhanced aivestor agent
        user_profile = agent1_for_workflow()
        
        print(f"✅ User profile created: {user_profile['profile_id']}")
        return user_profile
    
    async def _run_agent2(self, user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run Agent 2: Comprehensive Market Analysis
        """
        print("Running comprehensive market analysis...")
        
        analysis_results = {}
        
        # 1. Macro Analysis
        print("  📈 Analyzing macro economic conditions...")
        try:
            macro_analyzer = EnhancedMacroAnalyzer()
            macro_analysis = macro_analyzer.generate_analysis_report()
            macro_signals = macro_analyzer.export_macro_signals_json()
            analysis_results['macro'] = {
                'report': macro_analysis,
                'signals': macro_signals
            }
            print("  ✅ Macro analysis complete")
        except Exception as e:
            print(f"  ⚠️ Macro analysis failed: {e}")
            analysis_results['macro'] = {"error": str(e)}
        
        # 2. Stock Analysis
        print("  📊 Analyzing stock market...")
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            config = AgentConfig(llm_config=LLMConfig(api_key=api_key))
            stock_agent = StockAnalysisAgent(config)
            
            # Analyze based on user preferences
            symbols = self._select_stocks_for_user(user_profile)
            stock_analysis = await stock_agent.analyze_stocks(
                symbols=symbols, 
                macro_context=analysis_results.get('macro', {}).get('signals', {})
            )
            analysis_results['stocks'] = stock_analysis
            print("  ✅ Stock analysis complete")
        except Exception as e:
            print(f"  ⚠️ Stock analysis failed: {e}")
            analysis_results['stocks'] = {"error": str(e)}
        
        # 3. Bond Analysis
        print("  🏛️ Analyzing bond market...")
        try:
            bond_agent = BondAnalysisAgent()
            bond_symbols = self._select_bonds_for_user(user_profile)
            bond_analysis = await bond_agent.analyze_bonds(symbols=bond_symbols)
            analysis_results['bonds'] = bond_analysis
            print("  ✅ Bond analysis complete")
        except Exception as e:
            print(f"  ⚠️ Bond analysis failed: {e}")
            analysis_results['bonds'] = {"error": str(e)}
        
        # 4. Gold Analysis
        print("  🏆 Analyzing precious metals...")
        try:
            gold_agent = GoldAnalysisAgent()
            gold_symbols = ["GLD", "IAU", "GC=F", "GDX", "SGOL"]
            gold_analysis = await gold_agent.analyze_gold(symbols=gold_symbols)
            analysis_results['gold'] = gold_analysis
            print("  ✅ Gold analysis complete")
        except Exception as e:
            print(f"  ⚠️ Gold analysis failed: {e}")
            analysis_results['gold'] = {"error": str(e)}
        
        return analysis_results
    
    def _load_analysis_outputs(self) -> Dict[str, Any]:
        """
        Load the most recent analysis outputs from JSON files in analysis_outputs directory
        """
        print("Loading analysis outputs from JSON files...")
        
        analysis_data = {}
        output_dir = Path("analysis_outputs")
        
        if not output_dir.exists():
            logger.warning("analysis_outputs directory not found")
            return analysis_data
        
        # Load user profile
        try:
            user_files = list(output_dir.glob("user_profile_*.json"))
            if user_files:
                latest_user_file = max(user_files, key=lambda f: f.stat().st_mtime)
                with open(latest_user_file, 'r') as f:
                    analysis_data['user_profile'] = json.load(f)
                print(f"  ✅ Loaded user profile from {latest_user_file.name}")
            else:
                logger.warning("No user profile file found in analysis_outputs")
        except Exception as e:
            logger.error(f"Failed to load user profile: {e}")
        
        # Load stock analysis
        try:
            stock_files = list(output_dir.glob("stock_analysis_*.json"))
            if stock_files:
                latest_stock_file = max(stock_files, key=lambda f: f.stat().st_mtime)
                with open(latest_stock_file, 'r') as f:
                    analysis_data['stock_analysis'] = json.load(f)
                print(f"  ✅ Loaded stock analysis from {latest_stock_file.name}")
            else:
                logger.warning("No stock analysis file found in analysis_outputs")
        except Exception as e:
            logger.error(f"Failed to load stock analysis: {e}")
        
        # Load bond analysis
        try:
            bond_files = list(output_dir.glob("bond_analysis_*.json"))
            if bond_files:
                latest_bond_file = max(bond_files, key=lambda f: f.stat().st_mtime)
                with open(latest_bond_file, 'r') as f:
                    analysis_data['bond_analysis'] = json.load(f)
                print(f"  ✅ Loaded bond analysis from {latest_bond_file.name}")
            else:
                logger.warning("No bond analysis file found in analysis_outputs")
        except Exception as e:
            logger.error(f"Failed to load bond analysis: {e}")
        
        # Load gold analysis
        try:
            gold_files = list(output_dir.glob("gold_*.json"))
            if gold_files:
                latest_gold_file = max(gold_files, key=lambda f: f.stat().st_mtime)
                with open(latest_gold_file, 'r') as f:
                    analysis_data['gold_analysis'] = json.load(f)
                print(f"  ✅ Loaded gold analysis from {latest_gold_file.name}")
            else:
                logger.warning("No gold analysis file found in analysis_outputs")
        except Exception as e:
            logger.error(f"Failed to load gold analysis: {e}")
        
        # Load macro analysis
        try:
            macro_files = list(output_dir.glob("macro_analysis_*.json"))
            if macro_files:
                latest_macro_file = max(macro_files, key=lambda f: f.stat().st_mtime)
                with open(latest_macro_file, 'r') as f:
                    analysis_data['macro_analysis'] = json.load(f)
                print(f"  ✅ Loaded macro analysis from {latest_macro_file.name}")
            else:
                logger.warning("No macro analysis file found in analysis_outputs")
        except Exception as e:
            logger.error(f"Failed to load macro analysis: {e}")
        
        return analysis_data
    
    async def _run_agent3(self, user_profile: Dict[str, Any], analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run Agent 3: Professional Portfolio Construction using saved JSON outputs
        """
        print("Constructing optimal portfolio using professional portfolio agent...")
        
        # Load actual analysis outputs from JSON files
        print("\n  📁 Loading saved analysis outputs...")
        saved_analysis = self._load_analysis_outputs()
        
        # Use saved data if available, otherwise fall back to in-memory results
        user_data = saved_analysis.get('user_profile', user_profile)
        stock_data = saved_analysis.get('stock_analysis', analysis_results.get('stocks', {}))
        bond_data = saved_analysis.get('bond_analysis', analysis_results.get('bonds', {}))
        gold_data = saved_analysis.get('gold_analysis', analysis_results.get('gold', {}))
        macro_data = saved_analysis.get('macro_analysis', analysis_results.get('macro', {}))
        
        print(f"  📊 Using data sources:")
        print(f"    - User Profile: {'✅ JSON file' if 'user_profile' in saved_analysis else '⚠️ In-memory'}")
        print(f"    - Stock Analysis: {'✅ JSON file' if 'stock_analysis' in saved_analysis else '⚠️ In-memory'}")
        print(f"    - Bond Analysis: {'✅ JSON file' if 'bond_analysis' in saved_analysis else '⚠️ In-memory'}")
        print(f"    - Gold Analysis: {'✅ JSON file' if 'gold_analysis' in saved_analysis else '⚠️ In-memory'}")
        print(f"    - Macro Analysis: {'✅ JSON file' if 'macro_analysis' in saved_analysis else '⚠️ In-memory'}")
        
        try:
            print("\n  🔨 Running professional portfolio construction...")
            
            # Call the professional portfolio agent
            portfolio_recommendation = await self.portfolio_agent.construct_portfolio(
                user_profile=user_data,
                stock_analysis=stock_data,
                bond_analysis=bond_data,
                gold_analysis=gold_data,
                market_conditions=macro_data
            )
            
            # Convert the portfolio recommendation to dict for serialization
            if hasattr(portfolio_recommendation, '__dict__'):
                portfolio_dict = asdict(portfolio_recommendation) if hasattr(portfolio_recommendation, '__dataclass_fields__') else vars(portfolio_recommendation)
            else:
                portfolio_dict = portfolio_recommendation
            
            print("  ✅ Professional portfolio construction complete")
            
            # Save the portfolio recommendation
            portfolio_filename = f"portfolio_recommendation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            portfolio_path = f"analysis_outputs/{portfolio_filename}"
            
            with open(portfolio_path, 'w') as f:
                json.dump(portfolio_dict, f, indent=2, default=str)
            
            print(f"  💾 Portfolio saved to: {portfolio_filename}")
            
            return portfolio_dict
            
        except Exception as e:
            logger.error(f"Professional portfolio construction failed: {e}")
            print(f"  ❌ Portfolio construction failed: {e}")
            
            # Fall back to basic portfolio construction
            print("  🔄 Falling back to basic portfolio construction...")
            return await self._basic_portfolio_fallback(user_data, stock_data, bond_data, gold_data)
    
    async def _basic_portfolio_fallback(self, user_profile: Dict, stock_data: Dict, bond_data: Dict, gold_data: Dict) -> Dict[str, Any]:
        """
        Basic portfolio construction fallback if professional agent fails
        """
        
        risk_level = user_profile.get('risk_level', 'moderate')
        investment_amount = user_profile.get('investment_amount', 10000)
        
        # Basic portfolio allocation based on risk level
        if risk_level == 'aggressive':
            allocation = {'stocks': 0.70, 'bonds': 0.20, 'gold': 0.10}
        elif risk_level == 'conservative':
            allocation = {'stocks': 0.40, 'bonds': 0.50, 'gold': 0.10}
        else:  # moderate
            allocation = {'stocks': 0.60, 'bonds': 0.30, 'gold': 0.10}
        
        # Extract top recommendations from analysis
        top_stocks = self._extract_top_picks(stock_data, 'stocks')
        top_bonds = self._extract_top_picks(bond_data, 'bonds')
        top_gold = self._extract_top_picks(gold_data, 'gold')
        
        # Create basic analysis results structure for compatibility
        analysis_results = {
            'stocks': stock_data,
            'bonds': bond_data,
            'gold': gold_data
        }
        
        portfolio = {
            'allocation': allocation,
            'recommendations': {
                'stocks': top_stocks,
                'bonds': top_bonds,
                'gold': top_gold
            },
            'total_investment': investment_amount,
            'expected_return': self._calculate_expected_return(allocation, analysis_results),
            'risk_level': risk_level,
            'reasoning': self._generate_portfolio_reasoning(user_profile, analysis_results, allocation)
        }
        
        print("  ✅ Portfolio construction complete")
        return portfolio
    
    def _select_stocks_for_user(self, user_profile: Dict[str, Any]) -> List[str]:
        """Select appropriate stocks based on user profile"""
        risk_level = user_profile.get('risk_level', 'moderate')
        preferred_sectors = user_profile.get('preferred_sectors', [])
        
        # Base stock selection
        if risk_level == 'aggressive':
            base_stocks = ['NVDA', 'TSLA', 'AMD', 'SHOP', 'SQ']
        elif risk_level == 'conservative':
            base_stocks = ['AAPL', 'MSFT', 'JNJ', 'PG', 'V']
        else:
            base_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        
        # Add sector-specific stocks
        sector_stocks = {
            'Technology': ['AAPL', 'MSFT', 'GOOGL', 'NVDA'],
            'Healthcare': ['JNJ', 'UNH', 'PFE', 'ABBV'],
            'Financial': ['JPM', 'BAC', 'V', 'MA']
        }
        
        for sector in preferred_sectors:
            if sector in sector_stocks:
                base_stocks.extend(sector_stocks[sector])
        
        return list(set(base_stocks))  # Remove duplicates
    
    def _select_bonds_for_user(self, user_profile: Dict[str, Any]) -> List[str]:
        """Select appropriate bonds based on user profile"""
        risk_level = user_profile.get('risk_level', 'moderate')
        
        if risk_level == 'aggressive':
            return ['HYG', 'JNK', 'EMB']  # High yield, emerging markets
        elif risk_level == 'conservative':
            return ['SHY', 'IEF', 'BND', 'AGG']  # Safe government and aggregate bonds
        else:
            return ['IEF', 'LQD', 'HYG', 'TIP']  # Balanced mix
    
    def _extract_top_picks(self, analysis: Dict[str, Any], asset_type: str) -> List[Dict]:
        """Extract top picks from analysis results"""
        if 'error' in analysis:
            return []
        
        if asset_type == 'stocks':
            horizons = analysis.get('horizons', {})
            next_quarter = horizons.get('next_quarter', [])
            return next_quarter[:5]  # Top 5 stocks
        
        elif asset_type == 'bonds':
            horizons = analysis.get('horizons', {})
            next_quarter = horizons.get('next_quarter', [])
            return next_quarter[:3]  # Top 3 bonds
        
        elif asset_type == 'gold':
            horizons = analysis.get('horizons', {})
            next_quarter = horizons.get('next_quarter', [])
            return next_quarter[:2]  # Top 2 gold investments
        
        return []
    
    def _calculate_expected_return(self, allocation: Dict, analysis_results: Dict) -> float:
        """Calculate portfolio expected return"""
        # Simplified calculation
        base_returns = {'stocks': 0.08, 'bonds': 0.04, 'gold': 0.06}
        
        expected_return = sum(
            allocation[asset] * base_returns[asset] 
            for asset in allocation
        )
        
        return round(expected_return, 3)
    
    def _generate_portfolio_reasoning(self, user_profile: Dict, analysis_results: Dict, allocation: Dict) -> str:
        """Generate reasoning for portfolio construction"""
        risk_level = user_profile.get('risk_level', 'moderate')
        investment_goal = user_profile.get('investment_goal', 'wealth building')
        
        reasoning = f"""
Portfolio constructed for {risk_level} risk investor with {investment_goal} goal.

Allocation Reasoning:
- Stocks ({allocation['stocks']*100:.0f}%): Selected for growth potential based on current market analysis
- Bonds ({allocation['bonds']*100:.0f}%): Provides stability and income generation
- Gold ({allocation['gold']*100:.0f}%): Hedge against inflation and market volatility

Market Conditions Considered:
- Current macro economic environment
- Individual asset analysis and rankings
- Risk-adjusted return optimization
"""
        return reasoning.strip()
    
    def _generate_final_output(self) -> Dict[str, Any]:
        """Generate the final comprehensive recommendation"""
        portfolio = self.results['agent3_portfolio']
        user_profile = self.results['agent1_profile']
        
        final_output = {
            'session_id': self.session_id,
            'user_profile_summary': {
                'investment_amount': user_profile.get('investment_amount'),
                'risk_level': user_profile.get('risk_level'),
                'time_horizon': user_profile.get('time_horizon'),
                'investment_goal': user_profile.get('investment_goal')
            },
            'portfolio_allocation': portfolio.get('allocation'),
            'specific_recommendations': portfolio.get('recommendations'),
            'expected_annual_return': f"{portfolio.get('expected_return', 0)*100:.1f}%",
            'reasoning': portfolio.get('reasoning'),
            'implementation_plan': {
                'immediate_actions': [
                    "Review and approve the recommended allocation",
                    "Open investment accounts if needed",
                    "Execute trades according to allocation"
                ],
                'monitoring': [
                    "Rebalance quarterly or when allocation drifts >5%",
                    "Review annually or when life circumstances change"
                ]
            },
            'risk_warnings': [
                "Past performance does not guarantee future results",
                "All investments carry risk of loss",
                "Consider consulting with a financial advisor"
            ]
        }
        
        return final_output
    
    def _save_complete_results(self):
        """Save complete workflow results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"complete_investment_recommendation_{timestamp}.json"
        filepath = os.path.join("analysis_outputs", filename)
        
        try:
            with open(filepath, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            print(f"💾 Complete results saved to: {filepath}")
            logger.info(f"Complete workflow results saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
    
    async def run_portfolio_only_workflow(self) -> Dict[str, Any]:
        """
        Run only Agent 3 (portfolio construction) using existing JSON outputs
        This enables modular testing and re-running portfolio construction
        """
        print("🎯 AURITE AI PORTFOLIO CONSTRUCTION")
        print("=" * 60)
        print("Using existing analysis outputs for portfolio construction")
        print("=" * 60)
        
        try:
            # Load all saved analysis outputs
            print("\n📁 Loading saved analysis outputs...")
            saved_analysis = self._load_analysis_outputs()
            
            if not saved_analysis:
                print("❌ No analysis outputs found. Please run the complete workflow first.")
                return {"error": "No analysis outputs found"}
            
            # Check which data sources we have
            required_keys = ['user_profile', 'stock_analysis', 'bond_analysis', 'gold_analysis']
            missing_keys = [key for key in required_keys if key not in saved_analysis]
            
            if missing_keys:
                print(f"⚠️ Missing analysis data: {', '.join(missing_keys)}")
                print("Portfolio construction will proceed with available data.")
            
            # Run portfolio construction
            print("\n🎯 Running Portfolio Construction (Agent 3)")
            print("-" * 40)
            
            user_data = saved_analysis.get('user_profile', {})
            stock_data = saved_analysis.get('stock_analysis', {})
            bond_data = saved_analysis.get('bond_analysis', {})
            gold_data = saved_analysis.get('gold_analysis', {})
            macro_data = saved_analysis.get('macro_analysis', {})
            
            # Use professional portfolio agent
            portfolio_recommendation = await self.portfolio_agent.construct_portfolio(
                user_profile=user_data,
                stock_analysis=stock_data,
                bond_analysis=bond_data,
                gold_analysis=gold_data,
                market_conditions=macro_data
            )
            
            # Convert to dict and save
            if hasattr(portfolio_recommendation, '__dict__'):
                portfolio_dict = asdict(portfolio_recommendation) if hasattr(portfolio_recommendation, '__dataclass_fields__') else vars(portfolio_recommendation)
            else:
                portfolio_dict = portfolio_recommendation
            
            # Save the portfolio recommendation
            portfolio_filename = f"portfolio_recommendation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            portfolio_path = f"analysis_outputs/{portfolio_filename}"
            
            with open(portfolio_path, 'w') as f:
                json.dump(portfolio_dict, f, indent=2, default=str)
            
            print(f"✅ Portfolio saved to: {portfolio_filename}")
            print("\n✅ PORTFOLIO CONSTRUCTION COMPLETE!")
            print("=" * 60)
            
            return {
                'session_id': self.session_id,
                'timestamp': datetime.now().isoformat(),
                'portfolio': portfolio_dict,
                'data_sources': list(saved_analysis.keys()),
                'output_file': portfolio_filename
            }
            
        except Exception as e:
            logger.error(f"Portfolio-only workflow failed: {e}")
            return {"error": str(e), "session_id": self.session_id}

# ===================== MAIN EXECUTION =====================

async def main():
    """
    Main function to run the Aurite Investment Workflow
    """
    import sys
    
    workflow = AuriteInvestmentWorkflow()
    
    if len(sys.argv) > 1 and sys.argv[1] == "--portfolio-only":
        # Run only portfolio construction using existing JSON files
        result = await workflow.run_portfolio_only_workflow()
    else:
        # Run complete workflow
        result = await workflow.run_complete_workflow()
    
    if 'error' in result:
        print(f"\n❌ Workflow failed: {result['error']}")
        return 1
    else:
        print(f"\n🎉 Workflow completed successfully!")
        print(f"Session ID: {result['session_id']}")
        return 0

if __name__ == "__main__":
    import asyncio
    exit_code = asyncio.run(main())
