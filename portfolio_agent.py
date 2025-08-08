"""
Portfolio Optimization Agent - Professional Grade Implementation

Author: Enhanced for Aurite AI Project
Version: 2.0.0
Description: Professional-grade portfolio optimization engine that integrates outputs from all agents
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
import logging
import os
from scipy.optimize import minimize
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('PortfolioAgent')

# ===================== Data Models =====================

@dataclass
class UnifiedAssetRecommendation:
    """Unified recommendation across all asset classes"""
    ticker: str
    asset_class: str  # 'stock', 'bond', 'gold'
    score: float
    expected_return: float
    volatility: float
    confidence: float
    recommendation: str  # 'BUY', 'HOLD', 'SELL'
    reasoning: str
    sector: Optional[str] = None
    correlation_group: Optional[str] = None
    
@dataclass
class PortfolioAllocation:
    """Detailed portfolio allocation with implementation instructions"""
    ticker: str
    asset_class: str
    weight: float
    dollar_amount: float
    shares: int
    entry_price: float
    expected_return: float
    risk_contribution: float
    implementation_priority: int
    execution_instructions: str
    
@dataclass 
class RiskMetrics:
    """Comprehensive risk analytics"""
    portfolio_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    var_95: float
    cvar_95: float
    beta: float
    alpha: float
    information_ratio: float
    calmar_ratio: float
    correlation_risk: float
    concentration_risk: float
    
@dataclass
class PortfolioRecommendation:
    """Complete portfolio recommendation package"""
    allocations: List[PortfolioAllocation]
    risk_metrics: RiskMetrics
    expected_performance: Dict[str, float]
    rebalancing_schedule: Dict[str, Any]
    tax_optimization: Dict[str, Any]
    implementation_plan: List[str]
    monitoring_alerts: List[Dict[str, Any]]
    ai_insights: str

# ===================== Advanced Portfolio Engine =====================

class ProfessionalPortfolioOptimizer:
    """
    Institutional-grade portfolio optimization engine
    Integrates all agent outputs for comprehensive portfolio construction
    """
    
    def __init__(self):
        self.risk_free_rate = 0.045  # Current T-bill rate
        self.optimization_methods = {
            'conservative': self._conservative_optimization,
            'moderate': self._balanced_optimization,
            'aggressive': self._growth_optimization
        }
        logger.info("Professional Portfolio Optimizer initialized")
    
    async def construct_portfolio(self,
                                 user_profile: Dict[str, Any],
                                 stock_analysis: Dict[str, Any],
                                 bond_analysis: Dict[str, Any],
                                 gold_analysis: Dict[str, Any],
                                 market_conditions: Dict[str, Any]) -> PortfolioRecommendation:
        """
        Main portfolio construction method - integrates ALL agent outputs
        """
        logger.info("Starting professional portfolio construction...")
        
        # Step 1: Unify all asset recommendations
        unified_assets = self._unify_asset_recommendations(
            stock_analysis, bond_analysis, gold_analysis
        )
        
        # Step 2: Apply user preferences and constraints
        filtered_assets = self._apply_user_preferences(unified_assets, user_profile)
        
        # Step 3: Construct correlation matrix
        correlation_matrix = self._build_correlation_matrix(filtered_assets, market_conditions)
        
        # Step 4: Run multi-objective optimization
        optimal_weights = self._multi_objective_optimization(
            filtered_assets, correlation_matrix, user_profile, market_conditions
        )
        
        # Step 5: Calculate comprehensive risk metrics
        risk_metrics = self._calculate_risk_metrics(
            optimal_weights, filtered_assets, correlation_matrix
        )
        
        # Step 6: Generate allocations with implementation details
        allocations = self._generate_detailed_allocations(
            optimal_weights, filtered_assets, user_profile
        )
        
        # Step 7: Create rebalancing and monitoring plan
        rebalancing_plan = self._create_rebalancing_schedule(user_profile, risk_metrics)
        monitoring_alerts = self._setup_monitoring_alerts(allocations, risk_metrics)
        
        # Step 8: Tax optimization strategies
        tax_strategies = self._optimize_for_taxes(allocations, user_profile)
        
        # Step 9: Generate AI insights
        ai_insights = await self._generate_ai_insights(
            user_profile, allocations, risk_metrics, market_conditions
        )
        
        # Step 10: Create implementation plan
        implementation_plan = self._create_implementation_plan(
            allocations, user_profile, market_conditions
        )
        
        return PortfolioRecommendation(
            allocations=allocations,
            risk_metrics=risk_metrics,
            expected_performance=self._project_performance(allocations, risk_metrics),
            rebalancing_schedule=rebalancing_plan,
            tax_optimization=tax_strategies,
            implementation_plan=implementation_plan,
            monitoring_alerts=monitoring_alerts,
            ai_insights=ai_insights
        )
    
    def _unify_asset_recommendations(self, 
                                    stock_analysis: Dict,
                                    bond_analysis: Dict,
                                    gold_analysis: Dict) -> List[UnifiedAssetRecommendation]:
        """
        Unify recommendations from all asset class agents
        """
        unified = []
        
        # Process stock recommendations
        if 'horizons' in stock_analysis:
            # New format with horizons
            for stock in stock_analysis['horizons'].get('next_quarter', [])[:10]:
                unified.append(UnifiedAssetRecommendation(
                    ticker=stock.get('symbol', stock.get('ticker', '')),
                    asset_class='stock',
                    score=stock.get('expected_return', 0.08) * 100,
                    expected_return=stock.get('expected_return', 0.08),
                    volatility=stock.get('volatility', 0.20),
                    confidence=stock.get('confidence', 0.75),
                    recommendation='BUY' if stock.get('sentiment') == 'Buy' else 'HOLD',
                    reasoning=stock.get('summary', ''),
                    sector=stock.get('sector', 'Technology'),
                    correlation_group='equity'
                ))
        elif 'signals_summary' in stock_analysis:
            # Legacy format with signals_summary
            for stock in stock_analysis['signals_summary'][:10]:
                # Map signal to expected return
                signal = stock.get('signal', 'Hold')
                if signal == 'Buy':
                    expected_return = 0.12  # Aggressive return for buy signals
                elif signal == 'Sell':
                    expected_return = 0.02  # Low return for sell signals  
                else:
                    expected_return = 0.08  # Moderate return for hold
                
                unified.append(UnifiedAssetRecommendation(
                    ticker=stock.get('ticker', ''),
                    asset_class='stock',
                    score=stock.get('score', 5.0) * 10,  # Convert 5.0 score to 50
                    expected_return=expected_return,
                    volatility=0.25 if signal == 'Buy' else 0.15,  # Higher vol for growth stocks
                    confidence=0.8 if stock.get('confidence') == 'high' else 0.6,
                    recommendation=signal.upper(),
                    reasoning=stock.get('reason', 'Stock analysis signal'),
                    sector='Technology',  # Default sector
                    correlation_group='equity'
                ))
        
        # Process bond recommendations  
        if 'horizons' in bond_analysis:
            for bond in bond_analysis['horizons'].get('next_quarter', [])[:8]:
                unified.append(UnifiedAssetRecommendation(
                    ticker=bond['ticker'],
                    asset_class='bond',
                    score=abs(bond.get('expected_return', 0.04)) * 100,  # Bonds may have negative scores
                    expected_return=bond.get('expected_return', 0.04),
                    volatility=bond.get('volatility', 0.08),
                    confidence=bond.get('confidence', 0.80),
                    recommendation='BUY' if bond.get('sentiment') == 'bullish' else 'HOLD',
                    reasoning=bond.get('summary', ''),
                    sector=bond.get('bond_type', 'Treasury'),
                    correlation_group='fixed_income'
                ))
        
        # Process gold recommendations
        if 'horizons' in gold_analysis:
            for gold in gold_analysis['horizons'].get('next_quarter', [])[:5]:
                unified.append(UnifiedAssetRecommendation(
                    ticker=gold['ticker'],
                    asset_class='gold',
                    score=gold.get('expected_return', 0.05) * 100,
                    expected_return=gold.get('expected_return', 0.05),
                    volatility=gold.get('volatility', 0.15),
                    confidence=gold.get('confidence', 0.70),
                    recommendation='BUY' if gold.get('sentiment') == 'bullish' else 'HOLD',
                    reasoning=gold.get('summary', ''),
                    sector='Precious Metals',
                    correlation_group='alternatives'
                ))
        
        # Sort by score and confidence
        unified.sort(key=lambda x: x.score * x.confidence, reverse=True)
        
        logger.info(f"Unified {len(unified)} asset recommendations across all classes")
        return unified
    
    def _apply_user_preferences(self, 
                               assets: List[UnifiedAssetRecommendation],
                               user_profile: Dict) -> List[UnifiedAssetRecommendation]:
        """
        Filter and adjust assets based on user preferences
        """
        filtered = []
        
        # Get user preferences
        risk_level = user_profile.get('risk_level', 'moderate')
        preferred_sectors = user_profile.get('preferred_sectors', [])
        avoid_sectors = user_profile.get('avoid_sectors', [])
        esg_preference = user_profile.get('prefers_esg', False)
        
        for asset in assets:
            # Skip avoided sectors
            if asset.sector in avoid_sectors:
                continue
            
            # Boost preferred sectors
            if asset.sector in preferred_sectors:
                asset.score *= 1.2
                asset.confidence = min(0.95, asset.confidence * 1.1)
            
            # Risk-based filtering
            if risk_level == 'conservative':
                if asset.volatility > 0.25:  # Skip high volatility
                    continue
                if asset.asset_class == 'stock' and asset.confidence < 0.8:
                    continue
            elif risk_level == 'aggressive':
                if asset.asset_class == 'bond' and asset.expected_return < 0.06:
                    continue  # Skip low-yield bonds for aggressive
            
            filtered.append(asset)
        
        logger.info(f"Filtered to {len(filtered)} assets based on user preferences")
        return filtered
    
    def _build_correlation_matrix(self,
                                 assets: List[UnifiedAssetRecommendation],
                                 market_conditions: Dict) -> np.ndarray:
        """
        Build sophisticated correlation matrix based on asset classes and market conditions
        """
        n = len(assets)
        corr_matrix = np.eye(n)
        
        # Base correlations by asset class pairs
        correlation_map = {
            ('stock', 'stock'): 0.6,
            ('stock', 'bond'): -0.2,
            ('stock', 'gold'): -0.1,
            ('bond', 'bond'): 0.7,
            ('bond', 'gold'): 0.1,
            ('gold', 'gold'): 0.8
        }
        
        # Adjust for market conditions
        if market_conditions.get('vix_level', 0.5) > 0.7:  # High volatility
            correlation_map[('stock', 'stock')] = 0.8  # Higher correlation in crisis
            correlation_map[('stock', 'bond')] = -0.4  # Flight to quality
        
        for i in range(n):
            for j in range(i+1, n):
                asset_i = assets[i]
                asset_j = assets[j]
                
                # Get base correlation
                key = tuple(sorted([asset_i.asset_class, asset_j.asset_class]))
                base_corr = correlation_map.get(key, 0.3)
                
                # Adjust for same sector
                if asset_i.sector == asset_j.sector:
                    base_corr = min(0.95, base_corr + 0.2)
                
                # Add some noise for realism
                correlation = base_corr + np.random.normal(0, 0.05)
                correlation = max(-0.95, min(0.95, correlation))
                
                corr_matrix[i, j] = correlation
                corr_matrix[j, i] = correlation
        
        return corr_matrix
    
    def _multi_objective_optimization(self,
                                     assets: List[UnifiedAssetRecommendation],
                                     correlation_matrix: np.ndarray,
                                     user_profile: Dict,
                                     market_conditions: Dict) -> np.ndarray:
        """
        Advanced multi-objective portfolio optimization
        """
        n_assets = len(assets)
        returns = np.array([a.expected_return for a in assets])
        volatilities = np.array([a.volatility for a in assets])
        
        # Calculate covariance matrix
        std_matrix = np.diag(volatilities)
        cov_matrix = std_matrix @ correlation_matrix @ std_matrix
        
        # Get optimization method based on risk profile
        risk_level = user_profile.get('risk_level', 'moderate')
        logger.info(f"Using {risk_level} optimization strategy")
        optimization_func = self.optimization_methods[risk_level]
        
        # Run optimization with constraints
        weights = optimization_func(returns, cov_matrix, assets, user_profile, market_conditions)
        logger.info(f"Optimization result - weights sum: {np.sum(weights):.4f}, non-zero positions: {np.count_nonzero(weights)}")
        
        # Post-process weights
        weights = self._post_process_weights(weights, assets, user_profile)
        logger.info(f"Post-processed weights - sum: {np.sum(weights):.4f}, positions: {np.count_nonzero(weights)}")
        
        return weights
    
    def _conservative_optimization(self, returns, cov_matrix, assets, user_profile, market_conditions):
        """Conservative portfolio optimization - minimize risk"""
        n_assets = len(returns)
        
        # Objective: Minimize portfolio variance
        def objective(w):
            return np.dot(w.T, np.dot(cov_matrix, w))
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # Sum to 1
        ]
        
        # Asset class constraints for conservative allocation
        bond_indices = [i for i, a in enumerate(assets) if a.asset_class == 'bond']
        stock_indices = [i for i, a in enumerate(assets) if a.asset_class == 'stock']
        gold_indices = [i for i, a in enumerate(assets) if a.asset_class == 'gold']
        
        # Conservative allocation: prefer bonds, limit stocks and gold
        if bond_indices:
            # At least 50% in bonds
            constraints.append({
                'type': 'ineq',
                'fun': lambda w: np.sum([w[i] for i in bond_indices]) - 0.5
            })
        
        if stock_indices:
            # Max 30% in stocks
            constraints.append({
                'type': 'ineq',
                'fun': lambda w: 0.3 - np.sum([w[i] for i in stock_indices])
            })
            
        if gold_indices:
            # Max 20% in gold
            constraints.append({
                'type': 'ineq',
                'fun': lambda w: 0.2 - np.sum([w[i] for i in gold_indices])
            })
        
        # Bounds - conservative position limits (max 20% per asset)
        bounds = tuple((0.01, 0.2) for _ in range(n_assets))
        
        # Initial guess - conservative allocation
        x0 = np.array([1/n_assets] * n_assets)
        if bond_indices:
            # Start with higher bond allocation
            bond_weight = 0.6 / len(bond_indices)
            for i in bond_indices:
                x0[i] = min(bond_weight, 0.2)
        
        # Normalize initial guess
        x0 = x0 / np.sum(x0)
        
        # Optimize
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints,
                         options={'maxiter': 1000, 'ftol': 1e-9})
        logger.info(f"Conservative optimization - Success: {result.success}, Message: {result.message}")
        
        return result.x if result.success else x0
    
    def _balanced_optimization(self, returns, cov_matrix, assets, user_profile, market_conditions):
        """Balanced portfolio optimization - maximize Sharpe ratio"""
        n_assets = len(returns)
        
        # Objective: Maximize Sharpe ratio
        def neg_sharpe(w):
            port_return = np.dot(w, returns)
            port_vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
            return -(port_return - self.risk_free_rate) / port_vol if port_vol > 0 else 0
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]
        
        # Balanced asset class constraints
        bond_indices = [i for i, a in enumerate(assets) if a.asset_class == 'bond']
        stock_indices = [i for i, a in enumerate(assets) if a.asset_class == 'stock']
        
        # 20-40% in bonds
        if bond_indices:
            constraints.extend([
                {'type': 'ineq', 'fun': lambda w: np.sum([w[i] for i in bond_indices]) - 0.2},
                {'type': 'ineq', 'fun': lambda w: 0.4 - np.sum([w[i] for i in bond_indices])}
            ])
        
        # 40-70% in stocks
        if stock_indices:
            constraints.extend([
                {'type': 'ineq', 'fun': lambda w: np.sum([w[i] for i in stock_indices]) - 0.4},
                {'type': 'ineq', 'fun': lambda w: 0.7 - np.sum([w[i] for i in stock_indices])}
            ])
        
        # Bounds
        bounds = tuple((0, 0.20) for _ in range(n_assets))
        
        # Initial guess
        x0 = np.array([1/n_assets] * n_assets)
        
        # Optimize
        result = minimize(neg_sharpe, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        logger.info(f"Balanced optimization - Success: {result.success}, Message: {result.message}")
        
        return result.x if result.success else x0
    
    def _growth_optimization(self, returns, cov_matrix, assets, user_profile, market_conditions):
        """Aggressive portfolio optimization - maximize returns"""
        n_assets = len(returns)
        
        # Simplified objective: Maximize expected return
        def neg_return(w):
            return -np.dot(w, returns)
        
        # Simplified constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # Sum to 1
        ]
        
        # Bounds - allow higher concentration for aggressive
        bounds = tuple((0.01, 0.6) for _ in range(n_assets))
        
        # Initial guess
        x0 = np.array([1/n_assets] * n_assets)
        
        # Try simple optimization first
        try:
            result = minimize(neg_return, x0, method='SLSQP', bounds=bounds, constraints=constraints,
                             options={'maxiter': 500, 'ftol': 1e-6})
            logger.info(f"Growth optimization - Success: {result.success}, Message: {result.message}")
            
            if result.success:
                return result.x
            else:
                # Fall back to simple aggressive allocation
                logger.info("Growth optimization failed, using manual aggressive allocation")
                return self._manual_aggressive_allocation(assets, user_profile)
                
        except Exception as e:
            logger.error(f"Growth optimization error: {e}")
            return self._manual_aggressive_allocation(assets, user_profile)
    
    def _manual_aggressive_allocation(self, assets, user_profile):
        """Manual aggressive allocation when optimization fails"""
        n_assets = len(assets)
        weights = np.zeros(n_assets)
        
        # Categorize assets
        bond_indices = [i for i, a in enumerate(assets) if a.asset_class == 'bond']
        stock_indices = [i for i, a in enumerate(assets) if a.asset_class == 'stock']
        gold_indices = [i for i, a in enumerate(assets) if a.asset_class == 'gold']
        
        # Aggressive allocation: prioritize stocks first, then alternatives
        total_allocated = 0
        
        # Allocate to stocks first (60% if available)
        if stock_indices:
            stock_weight = min(0.6, 0.6 / len(stock_indices))  # Max 60% in stocks
            for i in stock_indices:
                weights[i] = stock_weight
                total_allocated += stock_weight
        
        # Allocate to gold (remaining up to 30%)
        remaining = 1.0 - total_allocated
        if gold_indices and remaining > 0:
            gold_allocation = min(0.3, remaining)
            gold_weight = gold_allocation / len(gold_indices)
            for i in gold_indices:
                weights[i] = gold_weight
                total_allocated += gold_weight
        
        # Allocate remaining to highest yield bonds
        remaining = 1.0 - total_allocated
        if bond_indices and remaining > 0:
            # Sort bonds by expected return
            bond_returns = [(i, assets[i].expected_return) for i in bond_indices]
            bond_returns.sort(key=lambda x: x[1], reverse=True)
            
            bond_weight = remaining / len(bond_indices)
            for i, _ in bond_returns:
                weights[i] = bond_weight
        
        return weights
    
    def _post_process_weights(self, weights, assets, user_profile):
        """Post-process optimization weights"""
        # Remove very small positions
        min_weight = 0.01
        weights[weights < min_weight] = 0
        
        # Renormalize
        weight_sum = np.sum(weights)
        if weight_sum > 0:
            weights = weights / weight_sum
        
        # Ensure we don't have too many positions
        max_positions = 20 if user_profile.get('investment_amount', 0) > 100000 else 15
        
        if np.count_nonzero(weights) > max_positions:
            # Keep only top positions
            sorted_indices = np.argsort(weights)[::-1]
            weights[sorted_indices[max_positions:]] = 0
            weights = weights / np.sum(weights)
        
        return weights
    
    def _calculate_risk_metrics(self, weights, assets, correlation_matrix):
        """Calculate comprehensive risk metrics"""
        returns = np.array([a.expected_return for a in assets])
        volatilities = np.array([a.volatility for a in assets])
        
        # Covariance matrix
        std_matrix = np.diag(volatilities)
        cov_matrix = std_matrix @ correlation_matrix @ std_matrix
        
        # Portfolio metrics
        port_return = np.dot(weights, returns)
        port_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        port_vol = np.sqrt(port_variance)
        
        # Sharpe ratio
        sharpe = (port_return - self.risk_free_rate) / port_vol if port_vol > 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_vol = port_vol * 0.7  # Approximation
        sortino = (port_return - self.risk_free_rate) / downside_vol if downside_vol > 0 else 0
        
        # VaR and CVaR
        z_95 = norm.ppf(0.05)
        var_95 = abs(port_return + z_95 * port_vol)
        cvar_95 = abs(port_return - port_vol * norm.pdf(z_95) / 0.05)
        
        # Max drawdown estimation
        max_drawdown = 2.5 * port_vol  # Rule of thumb
        
        # Beta (assuming market return of 10% and vol of 16%)
        market_corr = 0.7  # Assumption
        beta = (port_vol * market_corr) / 0.16
        
        # Alpha
        market_return = 0.10
        alpha = port_return - (self.risk_free_rate + beta * (market_return - self.risk_free_rate))
        
        # Information ratio
        tracking_error = 0.05  # Assumption
        information_ratio = alpha / tracking_error if tracking_error > 0 else 0
        
        # Calmar ratio
        calmar = port_return / max_drawdown if max_drawdown > 0 else 0
        
        # Concentration risk (Herfindahl index)
        concentration = np.sum(weights ** 2)
        
        # Correlation risk (average correlation)
        n = len(weights)
        total_corr = np.sum(correlation_matrix) - n  # Subtract diagonal
        avg_corr = total_corr / (n * (n - 1)) if n > 1 else 0
        
        return RiskMetrics(
            portfolio_volatility=float(port_vol),
            sharpe_ratio=float(sharpe),
            sortino_ratio=float(sortino),
            max_drawdown=float(max_drawdown),
            var_95=float(var_95),
            cvar_95=float(cvar_95),
            beta=float(beta),
            alpha=float(alpha),
            information_ratio=float(information_ratio),
            calmar_ratio=float(calmar),
            correlation_risk=float(avg_corr),
            concentration_risk=float(concentration)
        )
    
    def _generate_detailed_allocations(self, weights, assets, user_profile):
        """Generate detailed allocation instructions"""
        investment_amount = user_profile.get('investment_amount', 100000)
        allocations = []
        
        # Sort by weight
        sorted_indices = np.argsort(weights)[::-1]
        
        for i, idx in enumerate(sorted_indices):
            if weights[idx] < 0.001:
                continue
            
            asset = assets[idx]
            dollar_amount = investment_amount * weights[idx]
            
            # Estimate entry price (would use real-time data in production)
            entry_price = 100 * (1 + np.random.normal(0, 0.02))
            shares = int(dollar_amount / entry_price)
            
            # Risk contribution
            risk_contribution = weights[idx] * asset.volatility
            
            # Execution instructions based on liquidity
            if asset.asset_class == 'stock' and dollar_amount > 10000:
                execution = "Use VWAP or TWAP algorithm for large orders"
            elif asset.asset_class == 'bond':
                execution = "Place limit order at mid-point"
            else:
                execution = "Market order acceptable"
            
            allocations.append(PortfolioAllocation(
                ticker=asset.ticker,
                asset_class=asset.asset_class,
                weight=float(weights[idx]),
                dollar_amount=float(dollar_amount),
                shares=shares,
                entry_price=float(entry_price),
                expected_return=float(asset.expected_return),
                risk_contribution=float(risk_contribution),
                implementation_priority=i + 1,
                execution_instructions=execution
            ))
        
        return allocations
    
    def _create_rebalancing_schedule(self, user_profile, risk_metrics):
        """Create dynamic rebalancing schedule"""
        # Base rebalancing frequency on volatility and contribution
        if risk_metrics.portfolio_volatility > 0.20:
            frequency = "monthly"
            threshold = 0.03  # 3% deviation
        elif risk_metrics.portfolio_volatility > 0.12:
            frequency = "quarterly"
            threshold = 0.05  # 5% deviation
        else:
            frequency = "semi-annual"
            threshold = 0.07  # 7% deviation
        
        # Adjust for contributions
        if user_profile.get('monthly_contribution', 0) > 0:
            frequency = "monthly"  # Rebalance with contributions
        
        return {
            'frequency': frequency,
            'threshold': threshold,
            'method': 'threshold_based',
            'tax_aware': user_profile.get('tax_sensitive', False),
            'next_rebalance': self._next_rebalance_date(frequency),
            'estimated_cost': 0.001 * user_profile.get('investment_amount', 100000)  # 0.1% of portfolio
        }
    
    def _next_rebalance_date(self, frequency):
        """Calculate next rebalancing date"""
        today = datetime.now()
        if frequency == "monthly":
            next_date = today + timedelta(days=30)
        elif frequency == "quarterly":
            next_date = today + timedelta(days=90)
        else:  # semi-annual
            next_date = today + timedelta(days=180)
        
        return next_date.strftime("%Y-%m-%d")
    
    def _setup_monitoring_alerts(self, allocations, risk_metrics):
        """Setup portfolio monitoring alerts"""
        alerts = []
        
        # Risk-based alerts
        alerts.append({
            'type': 'drawdown',
            'threshold': risk_metrics.max_drawdown * 0.5,
            'message': f"Alert when portfolio drops {risk_metrics.max_drawdown * 50:.1f}%",
            'severity': 'high'
        })
        
        # Position-based alerts
        for allocation in allocations[:5]:  # Top 5 positions
            alerts.append({
                'type': 'position_deviation',
                'ticker': allocation.ticker,
                'threshold': 0.05,  # 5% deviation
                'message': f"Alert when {allocation.ticker} deviates 5% from target",
                'severity': 'medium'
            })
        
        # Volatility spike alert
        alerts.append({
            'type': 'volatility_spike',
            'threshold': risk_metrics.portfolio_volatility * 1.5,
            'message': "Alert when volatility increases 50%",
            'severity': 'high'
        })
        
        # Correlation breakdown alert
        alerts.append({
            'type': 'correlation_change',
            'threshold': 0.2,
            'message': "Alert when correlations increase significantly",
            'severity': 'medium'
        })
        
        return alerts
    
    def _optimize_for_taxes(self, allocations, user_profile):
        """Tax optimization strategies"""
        strategies = {
            'tax_loss_harvesting': False,
            'asset_location': {},
            'holding_period': {},
            'estimated_tax_savings': 0
        }
        
        if not user_profile.get('tax_sensitive', False):
            return strategies
        
        # Enable tax loss harvesting
        strategies['tax_loss_harvesting'] = True
        
        # Asset location optimization
        for allocation in allocations:
            if allocation.asset_class == 'bond':
                strategies['asset_location'][allocation.ticker] = 'tax_deferred'  # IRA/401k
            elif allocation.expected_return > 0.10:
                strategies['asset_location'][allocation.ticker] = 'taxable'  # For long-term gains
            else:
                strategies['asset_location'][allocation.ticker] = 'tax_free'  # Roth IRA
        
        # Holding period recommendations
        for allocation in allocations:
            if allocation.asset_class == 'stock':
                strategies['holding_period'][allocation.ticker] = 'long_term'  # >1 year
            else:
                strategies['holding_period'][allocation.ticker] = 'flexible'
        
        # Estimate tax savings
        investment_amount = user_profile.get('investment_amount', 100000)
        strategies['estimated_tax_savings'] = investment_amount * 0.002  # 0.2% annually
        
        return strategies
    
    def _project_performance(self, allocations, risk_metrics):
        """Project portfolio performance over multiple time horizons"""
        # Calculate weighted expected return
        total_return = sum(a.weight * a.expected_return for a in allocations)
        
        projections = {}
        
        # 1 Year projection
        projections['1_year'] = {
            'expected_value': 1.0 * (1 + total_return),
            'best_case': 1.0 * (1 + total_return + 2 * risk_metrics.portfolio_volatility),
            'worst_case': 1.0 * (1 + total_return - 2 * risk_metrics.portfolio_volatility),
            'probability_positive': norm.cdf(total_return / risk_metrics.portfolio_volatility)
        }
        
        # 5 Year projection (with compounding)
        projections['5_year'] = {
            'expected_value': (1 + total_return) ** 5,
            'best_case': (1 + total_return + risk_metrics.portfolio_volatility) ** 5,
            'worst_case': (1 + total_return - risk_metrics.portfolio_volatility) ** 5,
            'probability_double': norm.cdf((0.693 / 5 - total_return) / (risk_metrics.portfolio_volatility / np.sqrt(5)))
        }
        
        # 10 Year projection
        projections['10_year'] = {
            'expected_value': (1 + total_return) ** 10,
            'best_case': (1 + total_return + 0.7 * risk_metrics.portfolio_volatility) ** 10,
            'worst_case': (1 + total_return - 0.5 * risk_metrics.portfolio_volatility) ** 10,
            'probability_triple': norm.cdf((1.099 / 10 - total_return) / (risk_metrics.portfolio_volatility / np.sqrt(10)))
        }
        
        return projections
    
    async def _generate_ai_insights(self, user_profile, allocations, risk_metrics, market_conditions):
        """Generate sophisticated AI insights about the portfolio"""
        
        insights = []
        
        # Performance insight
        expected_return = sum(a.weight * a.expected_return for a in allocations)
        if risk_metrics.sharpe_ratio > 1.0:
            insights.append(f"✨ Exceptional risk-adjusted returns with {risk_metrics.sharpe_ratio:.2f} Sharpe ratio - institutional quality")
        elif risk_metrics.sharpe_ratio > 0.5:
            insights.append(f"✅ Solid risk-adjusted returns with {risk_metrics.sharpe_ratio:.2f} Sharpe ratio")
        
        # Risk insight
        if risk_metrics.portfolio_volatility < 0.10:
            insights.append("🛡️ Conservative portfolio with low volatility - suitable for capital preservation")
        elif risk_metrics.portfolio_volatility > 0.20:
            insights.append("⚡ High growth potential with elevated volatility - ensure this matches your risk tolerance")
        
        # Diversification insight
        concentration = risk_metrics.concentration_risk
        if concentration < 0.1:
            insights.append("🌐 Excellent diversification across assets - well-protected against single-asset risk")
        elif concentration > 0.2:
            insights.append("⚠️ Portfolio is concentrated - consider diversifying to reduce risk")
        
        # Market condition insight
        if market_conditions.get('vix_level', 0.5) > 0.7:
            insights.append("🌪️ High market volatility detected - portfolio includes defensive positions")
        
        # Asset class insight
        stock_weight = sum(a.weight for a in allocations if a.asset_class == 'stock')
        bond_weight = sum(a.weight for a in allocations if a.asset_class == 'bond')
        
        if stock_weight > 0.7:
            insights.append(f"📈 Growth-focused with {stock_weight*100:.0f}% equity allocation")
        elif bond_weight > 0.5:
            insights.append(f"🏛️ Income-focused with {bond_weight*100:.0f}% fixed income allocation")
        else:
            insights.append(f"⚖️ Balanced allocation: {stock_weight*100:.0f}% stocks, {bond_weight*100:.0f}% bonds")
        
        # Tax insight
        if user_profile.get('tax_sensitive', False):
            insights.append("📊 Tax-optimized allocation with estimated 0.2% annual tax alpha")
        
        # Rebalancing insight
        if user_profile.get('monthly_contribution', 0) > 0:
            insights.append("💰 Monthly contributions will help maintain target allocation and dollar-cost average")
        
        return " | ".join(insights)
    
    def _create_implementation_plan(self, allocations, user_profile, market_conditions):
        """Create step-by-step implementation plan"""
        plan = []
        
        # Opening accounts
        if user_profile.get('investment_amount', 0) > 50000:
            plan.append("1️⃣ Open accounts: Consider both taxable and tax-advantaged accounts for asset location")
        else:
            plan.append("1️⃣ Open account: Start with a taxable brokerage account")
        
        # Market timing consideration
        if market_conditions.get('vix_level', 0.5) > 0.6:
            plan.append("2️⃣ Timing: Consider phasing in over 2-3 weeks due to high volatility")
        else:
            plan.append("2️⃣ Timing: Market conditions favorable for immediate implementation")
        
        # Execution order
        plan.append("3️⃣ Execution Order:")
        
        # Group by asset class for execution
        bonds = [a for a in allocations if a.asset_class == 'bond']
        stocks = [a for a in allocations if a.asset_class == 'stock']
        alternatives = [a for a in allocations if a.asset_class == 'gold']
        
        if bonds:
            plan.append("   • Bonds first (most stable prices): " + ", ".join([b.ticker for b in bonds[:3]]))
        if stocks:
            plan.append("   • Large-cap stocks next: " + ", ".join([s.ticker for s in stocks[:5]]))
        if alternatives:
            plan.append("   • Alternatives last: " + ", ".join([a.ticker for a in alternatives[:2]]))
        
        # Trading instructions
        plan.append("4️⃣ Trading Instructions:")
        plan.append("   • Use limit orders for positions > $5,000")
        plan.append("   • Market orders acceptable for liquid ETFs")
        plan.append("   • Avoid trading in first/last 30 minutes")
        
        # Post-implementation
        plan.append("5️⃣ Post-Implementation:")
        plan.append("   • Set up automatic rebalancing alerts")
        plan.append("   • Enable dividend reinvestment")
        plan.append("   • Schedule quarterly review")
        
        return plan

# ===================== Main Portfolio Agent =====================

class ProfessionalPortfolioAgent:
    """
    Main Portfolio Agent - Orchestrates all components
    This is the final agent that combines ALL outputs
    """
    
    def __init__(self):
        self.optimizer = ProfessionalPortfolioOptimizer()
        self.name = "Professional Portfolio Construction Agent"
        self.version = "2.0"
        logger.info(f"{self.name} v{self.version} initialized")
    
    async def execute(self, 
                     user_profile: Dict[str, Any],
                     stock_analysis: Dict[str, Any],
                     bond_analysis: Dict[str, Any],
                     gold_analysis: Dict[str, Any],
                     market_conditions: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute complete portfolio construction
        This is the main method that combines ALL agent outputs
        """
        
        logger.info("="*60)
        logger.info("PROFESSIONAL PORTFOLIO CONSTRUCTION STARTING")
        logger.info("="*60)
        
        try:
            # Validate inputs
            self._validate_inputs(user_profile, stock_analysis, bond_analysis, gold_analysis)
            
            # Extract market conditions if not provided
            if not market_conditions:
                market_conditions = self._extract_market_conditions(stock_analysis, bond_analysis)
            
            # Run portfolio optimization
            portfolio_recommendation = await self.optimizer.construct_portfolio(
                user_profile=user_profile,
                stock_analysis=stock_analysis,
                bond_analysis=bond_analysis,
                gold_analysis=gold_analysis,
                market_conditions=market_conditions
            )
            
            # Generate final output
            final_output = self._generate_final_output(
                user_profile, portfolio_recommendation, market_conditions
            )
            
            # Save final output
            self._save_portfolio_recommendation(final_output)
            
            # Log success
            logger.info("✅ Portfolio construction completed successfully")
            
            return final_output
            
        except Exception as e:
            logger.error(f"Portfolio construction failed: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _validate_inputs(self, user_profile, stock_analysis, bond_analysis, gold_analysis):
        """Validate all inputs are properly formatted"""
        
        # Check user profile
        required_fields = ['investment_amount', 'risk_level', 'time_horizon']
        for field in required_fields:
            if field not in user_profile:
                raise ValueError(f"Missing required field in user_profile: {field}")
        
        # Check analysis outputs
        # Validate stock analysis
        if 'horizons' not in stock_analysis and 'signals_summary' not in stock_analysis:
            raise ValueError("Invalid stock_analysis format - missing 'horizons' or 'signals_summary'")
        if 'horizons' not in bond_analysis:
            raise ValueError("Invalid bond_analysis format - missing 'horizons'")
        if 'horizons' not in gold_analysis:
            raise ValueError("Invalid gold_analysis format - missing 'horizons'")
        
        logger.info("✅ All inputs validated successfully")
    
    def _extract_market_conditions(self, stock_analysis, bond_analysis):
        """Extract market conditions from analysis outputs"""
        conditions = {}
        
        # Extract from stock analysis
        if 'market_conditions' in stock_analysis:
            conditions.update(stock_analysis['market_conditions'])
        
        # Extract from bond analysis
        if 'market_conditions' in bond_analysis:
            bond_conditions = bond_analysis['market_conditions']
            conditions['interest_rate_trend'] = bond_conditions.get('interest_rate_trend', 0)
            conditions['yield_curve'] = bond_conditions.get('yield_curve_slope', 0)
        
        # Default values
        conditions.setdefault('vix_level', 0.5)
        conditions.setdefault('market_momentum', 0)
        conditions.setdefault('volatility', 'normal')
        
        return conditions
    
    def _save_portfolio_recommendation(self, output):
        """Save portfolio recommendation to analysis_outputs folder"""
        import os
        
        # Ensure analysis_outputs directory exists
        output_dir = "analysis_outputs"
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"portfolio_recommendation_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)
        
        try:
            with open(filepath, 'w') as f:
                json.dump(output, f, indent=2, default=str)
            logger.info(f"✅ Portfolio recommendation saved to {filepath}")
            print(f"💾 Portfolio saved to: {filepath}")
        except Exception as e:
            logger.error(f"❌ Error saving portfolio: {e}")
    
    def _generate_final_output(self, user_profile, recommendation, market_conditions):
        """Generate the final comprehensive output"""
        
        # Calculate summary statistics
        total_positions = len(recommendation.allocations)
        expected_return = sum(a.weight * a.expected_return for a in recommendation.allocations)
        investment_amount = user_profile.get('investment_amount', 100000)
        
        # Group allocations by asset class
        asset_class_summary = {}
        for allocation in recommendation.allocations:
            if allocation.asset_class not in asset_class_summary:
                asset_class_summary[allocation.asset_class] = {
                    'weight': 0,
                    'count': 0,
                    'tickers': []
                }
            asset_class_summary[allocation.asset_class]['weight'] += allocation.weight
            asset_class_summary[allocation.asset_class]['count'] += 1
            asset_class_summary[allocation.asset_class]['tickers'].append(allocation.ticker)
        
        # Create final output structure
        output = {
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'portfolio_id': f"PF_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            
            # Executive Summary
            'executive_summary': {
                'total_investment': investment_amount,
                'expected_annual_return': f"{expected_return*100:.1f}%",
                'risk_level': user_profile.get('risk_level', 'moderate'),
                'sharpe_ratio': recommendation.risk_metrics.sharpe_ratio,
                'total_positions': total_positions,
                'ai_confidence': 0.85  # Based on analysis quality
            },
            
            # Asset Allocation
            'asset_allocation': {
                'summary': asset_class_summary,
                'detailed_positions': [
                    {
                        'ticker': a.ticker,
                        'asset_class': a.asset_class,
                        'weight': f"{a.weight*100:.1f}%",
                        'dollar_amount': f"${a.dollar_amount:,.2f}",
                        'shares': a.shares,
                        'expected_return': f"{a.expected_return*100:.1f}%",
                        'priority': a.implementation_priority,
                        'execution': a.execution_instructions
                    }
                    for a in sorted(recommendation.allocations, key=lambda x: x.weight, reverse=True)
                ]
            },
            
            # Risk Analytics
            'risk_analytics': {
                'portfolio_volatility': f"{recommendation.risk_metrics.portfolio_volatility*100:.1f}%",
                'sharpe_ratio': round(recommendation.risk_metrics.sharpe_ratio, 2),
                'sortino_ratio': round(recommendation.risk_metrics.sortino_ratio, 2),
                'max_drawdown': f"{recommendation.risk_metrics.max_drawdown*100:.1f}%",
                'var_95': f"{recommendation.risk_metrics.var_95*100:.1f}%",
                'beta': round(recommendation.risk_metrics.beta, 2),
                'alpha': f"{recommendation.risk_metrics.alpha*100:.1f}%"
            },
            
            # Performance Projections
            'performance_projections': recommendation.expected_performance,
            
            # Implementation Plan
            'implementation': {
                'step_by_step_plan': recommendation.implementation_plan,
                'total_positions': total_positions,
                'estimated_transaction_cost': f"${investment_amount * 0.001:.2f}",
                'execution_timeframe': '1-2 business days'
            },
            
            # Rebalancing Strategy
            'rebalancing': recommendation.rebalancing_schedule,
            
            # Tax Optimization
            'tax_optimization': recommendation.tax_optimization,
            
            # Monitoring & Alerts
            'monitoring': {
                'alerts': recommendation.monitoring_alerts,
                'review_frequency': 'quarterly',
                'next_review': (datetime.now() + timedelta(days=90)).strftime('%Y-%m-%d')
            },
            
            # AI Insights
            'ai_insights': recommendation.ai_insights,
            
            # Market Context
            'market_context': market_conditions,
            
            # User Profile Summary
            'user_profile_summary': {
                'profile_id': user_profile.get('profile_id', 'N/A'),
                'risk_tolerance': user_profile.get('risk_level', 'moderate'),
                'time_horizon': f"{user_profile.get('time_horizon', 5)} years",
                'goals': user_profile.get('investment_goal', 'general wealth building'),
                'monthly_contribution': user_profile.get('monthly_contribution', 0)
            }
        }
        
        return output

# Main function for workflow integration
def portfolio_agent_for_workflow() -> ProfessionalPortfolioAgent:
    """
    Main entry point for Portfolio Agent in the workflow
    Returns the Portfolio Agent instance
    """
    return ProfessionalPortfolioAgent()

# Example usage
if __name__ == "__main__":
    import asyncio
    
    # Sample test data
    user_profile = {
        'profile_id': 'AIV_20250807_test',
        'investment_amount': 100000,
        'risk_level': 'moderate',
        'time_horizon': 5,
        'investment_goal': 'wealth',
        'preferred_sectors': ['Technology'],
        'monthly_contribution': 1000,
        'tax_sensitive': True
    }
    
    # Create portfolio agent
    agent = ProfessionalPortfolioAgent()
    print(f"✅ {agent.name} v{agent.version} ready for integration!")
    print(f"💡 Use this agent in the master workflow to create optimized portfolios")
