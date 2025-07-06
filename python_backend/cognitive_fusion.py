"""
Cognitive State Fusion Engine - Advanced multimodal analysis and state synthesis
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from enum import Enum
import json
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
from datetime import datetime

logger = logging.getLogger(__name__)

class CognitiveState(Enum):
    """Enumeration of cognitive states"""
    CALM = "calm"
    FOCUSED = "focused"
    ANXIOUS = "anxious"
    STRESSED = "stressed"
    FATIGUED = "fatigued"
    EXCITED = "excited"
    NEUTRAL = "neutral"

@dataclass
class ModalityResult:
    """Result from a single modality analysis"""
    modality: str
    confidence: float
    features: Dict[str, float]
    raw_predictions: Dict[str, float]
    quality_score: float
    processing_time_ms: float

@dataclass
class CognitiveStateResult:
    """Final cognitive state analysis result"""
    state: CognitiveState
    confidence: float
    valence: float  # -1 (negative) to +1 (positive)
    arousal: float  # 0 (calm) to 1 (excited)
    attention: float  # 0 (unfocused) to 1 (highly focused)
    stress_level: float  # 0 (no stress) to 1 (high stress)
    insights: List[str]
    recommendations: List[str]
    modality_contributions: Dict[str, float]
    graph_data: Optional[str] = None  # Base64 encoded graph

class CognitiveFusionEngine:
    """Advanced fusion engine for multimodal cognitive state analysis"""
    
    def __init__(self):
        # Cognitive state mappings
        self.state_mappings = {
            CognitiveState.CALM: {"valence": 0.3, "arousal": 0.2, "stress": 0.1},
            CognitiveState.FOCUSED: {"valence": 0.4, "arousal": 0.6, "stress": 0.2},
            CognitiveState.ANXIOUS: {"valence": -0.4, "arousal": 0.7, "stress": 0.8},
            CognitiveState.STRESSED: {"valence": -0.6, "arousal": 0.8, "stress": 0.9},
            CognitiveState.FATIGUED: {"valence": -0.2, "arousal": 0.1, "stress": 0.3},
            CognitiveState.EXCITED: {"valence": 0.8, "arousal": 0.9, "stress": 0.3},
            CognitiveState.NEUTRAL: {"valence": 0.0, "arousal": 0.5, "stress": 0.4}
        }
        
        # Emotion to cognitive state mappings
        self.emotion_to_cognitive = {
            'happy': [CognitiveState.EXCITED, CognitiveState.CALM],
            'surprise': [CognitiveState.EXCITED, CognitiveState.FOCUSED],
            'neutral': [CognitiveState.NEUTRAL, CognitiveState.CALM],
            'sad': [CognitiveState.FATIGUED, CognitiveState.STRESSED],
            'angry': [CognitiveState.STRESSED, CognitiveState.ANXIOUS],
            'fear': [CognitiveState.ANXIOUS, CognitiveState.STRESSED],
            'disgust': [CognitiveState.STRESSED, CognitiveState.ANXIOUS]
        }
        
        # Modality weights (can be adjusted based on quality)
        self.base_weights = {
            'emotion': 0.4,
            'audio': 0.3,
            'motion': 0.3
        }
        
    def fuse_modalities(self, modality_results: List[ModalityResult]) -> CognitiveStateResult:
        """Fuse results from multiple modalities into cognitive state"""
        try:
            # Calculate adaptive weights based on quality
            weights = self._calculate_adaptive_weights(modality_results)
            
            # Extract key metrics from each modality
            valence_scores = []
            arousal_scores = []
            stress_scores = []
            attention_scores = []
            
            modality_contributions = {}
            
            for result in modality_results:
                weight = weights.get(result.modality, 0.0)
                modality_contributions[result.modality] = weight
                
                if result.modality == 'emotion':
                    v, a, s, att = self._extract_emotion_metrics(result)
                elif result.modality == 'audio':
                    v, a, s, att = self._extract_audio_metrics(result)
                elif result.modality == 'motion':
                    v, a, s, att = self._extract_motion_metrics(result)
                else:
                    continue
                
                valence_scores.append((v, weight))
                arousal_scores.append((a, weight))
                stress_scores.append((s, weight))
                attention_scores.append((att, weight))
            
            # Weighted fusion
            final_valence = self._weighted_average(valence_scores)
            final_arousal = self._weighted_average(arousal_scores)
            final_stress = self._weighted_average(stress_scores)
            final_attention = self._weighted_average(attention_scores)
            
            # Determine cognitive state
            cognitive_state, confidence = self._determine_cognitive_state(
                final_valence, final_arousal, final_stress, final_attention
            )
            
            # Generate insights and recommendations
            insights = self._generate_insights(
                cognitive_state, final_valence, final_arousal, final_stress, final_attention, modality_results
            )
            recommendations = self._generate_recommendations(cognitive_state, modality_results)
            
            # Generate visualization
            graph_data = self._generate_graph(
                final_valence, final_arousal, final_stress, final_attention, modality_results
            )
            
            return CognitiveStateResult(
                state=cognitive_state,
                confidence=confidence,
                valence=final_valence,
                arousal=final_arousal,
                attention=final_attention,
                stress_level=final_stress,
                insights=insights,
                recommendations=recommendations,
                modality_contributions=modality_contributions,
                graph_data=graph_data
            )
            
        except Exception as e:
            logger.error(f"Error in modality fusion: {e}")
            return self._default_cognitive_result()
    
    def _calculate_adaptive_weights(self, modality_results: List[ModalityResult]) -> Dict[str, float]:
        """Calculate adaptive weights based on modality quality and availability"""
        weights = {}
        total_quality = 0
        
        # Calculate quality-weighted contributions
        for result in modality_results:
            base_weight = self.base_weights.get(result.modality, 0.0)
            quality_factor = result.quality_score
            confidence_factor = result.confidence
            
            # Combined quality score
            combined_quality = (quality_factor * 0.6 + confidence_factor * 0.4)
            weighted_score = base_weight * combined_quality
            
            weights[result.modality] = weighted_score
            total_quality += weighted_score
        
        # Normalize weights
        if total_quality > 0:
            weights = {k: v / total_quality for k, v in weights.items()}
        else:
            # Equal weights if no quality info
            n_modalities = len(modality_results)
            weights = {r.modality: 1.0 / n_modalities for r in modality_results}
        
        return weights
    
    def _extract_emotion_metrics(self, result: ModalityResult) -> Tuple[float, float, float, float]:
        """Extract valence, arousal, stress, attention from emotion analysis"""
        emotions = result.raw_predictions
        
        # Calculate valence (positive/negative emotions)
        positive_emotions = emotions.get('happy', 0) + emotions.get('surprise', 0) * 0.5
        negative_emotions = emotions.get('sad', 0) + emotions.get('angry', 0) + emotions.get('fear', 0) + emotions.get('disgust', 0)
        valence = positive_emotions - negative_emotions
        
        # Calculate arousal (high energy emotions)
        high_arousal = emotions.get('angry', 0) + emotions.get('fear', 0) + emotions.get('surprise', 0) + emotions.get('happy', 0) * 0.7
        low_arousal = emotions.get('sad', 0) + emotions.get('neutral', 0)
        arousal = high_arousal - low_arousal * 0.5
        
        # Calculate stress (negative high-arousal emotions)
        stress = emotions.get('angry', 0) + emotions.get('fear', 0) + emotions.get('disgust', 0) * 0.7
        
        # Calculate attention (focused vs distracted indicators)
        # Neutral and slightly positive emotions indicate good attention
        attention = emotions.get('neutral', 0) * 0.8 + emotions.get('happy', 0) * 0.6 - emotions.get('sad', 0) * 0.3
        
        return np.clip(valence, -1, 1), np.clip(arousal, 0, 1), np.clip(stress, 0, 1), np.clip(attention, 0, 1)
    
    def _extract_audio_metrics(self, result: ModalityResult) -> Tuple[float, float, float, float]:
        """Extract valence, arousal, stress, attention from audio analysis"""
        features = result.features
        
        # Map audio features to psychological metrics
        # High stress typically correlates with:
        # - Higher pitch variance
        # - Faster speaking rate
        # - Higher spectral energy in stress-related frequencies
        
        stress_level = result.raw_predictions.get('stress_level', 0.5)
        
        # Estimate other metrics from stress and audio features
        valence = 1.0 - stress_level  # High stress = low valence
        arousal = min(stress_level * 1.5, 1.0)  # Stress increases arousal
        attention = max(0.8 - stress_level, 0.2)  # High stress reduces attention
        
        # Adjust based on specific audio features if available
        if 'energy' in features:
            arousal = (arousal + features['energy']) / 2
        if 'pitch_stability' in features:
            attention = (attention + features['pitch_stability']) / 2
        
        return np.clip(valence, -1, 1), np.clip(arousal, 0, 1), np.clip(stress_level, 0, 1), np.clip(attention, 0, 1)
    
    def _extract_motion_metrics(self, result: ModalityResult) -> Tuple[float, float, float, float]:
        """Extract valence, arousal, stress, attention from motion analysis"""
        features = result.features
        
        # Motion patterns for psychological metrics
        stress_level = features.get('stress_level', 0.5)
        motion_energy = features.get('motion_energy', 0.5)
        stability = features.get('stability', 0.5)
        
        # Calculate metrics
        valence = max(0.2, 1.0 - stress_level * 0.8)  # Less stress = higher valence
        arousal = min(motion_energy, 1.0)  # Higher motion = higher arousal
        attention = stability  # More stable motion = better attention
        
        return np.clip(valence, -1, 1), np.clip(arousal, 0, 1), np.clip(stress_level, 0, 1), np.clip(attention, 0, 1)
    
    def _weighted_average(self, scores_and_weights: List[Tuple[float, float]]) -> float:
        """Calculate weighted average"""
        if not scores_and_weights:
            return 0.5
        
        total_weight = sum(weight for _, weight in scores_and_weights)
        if total_weight == 0:
            return np.mean([score for score, _ in scores_and_weights])
        
        weighted_sum = sum(score * weight for score, weight in scores_and_weights)
        return weighted_sum / total_weight
    
    def _determine_cognitive_state(self, valence: float, arousal: float, stress: float, attention: float) -> Tuple[CognitiveState, float]:
        """Determine cognitive state from psychological metrics"""
        
        # Calculate distances to each cognitive state
        state_distances = {}
        
        for state, target_metrics in self.state_mappings.items():
            # Calculate Euclidean distance in psychological space
            distance = np.sqrt(
                (valence - target_metrics['valence']) ** 2 +
                (arousal - target_metrics['arousal']) ** 2 +
                (stress - target_metrics['stress']) ** 2
            )
            state_distances[state] = distance
        
        # Find closest state
        best_state = min(state_distances.keys(), key=lambda s: state_distances[s])
        min_distance = state_distances[best_state]
        
        # Calculate confidence (closer = higher confidence)
        max_possible_distance = np.sqrt(3)  # Maximum distance in our 3D space
        confidence = max(0.1, 1.0 - (min_distance / max_possible_distance))
        
        # Apply some heuristic rules for edge cases
        if stress > 0.7:
            if arousal > 0.6:
                best_state = CognitiveState.ANXIOUS
            else:
                best_state = CognitiveState.STRESSED
        elif valence > 0.6 and arousal > 0.7:
            best_state = CognitiveState.EXCITED
        elif attention > 0.7 and stress < 0.3:
            best_state = CognitiveState.FOCUSED
        elif arousal < 0.3 and stress < 0.4:
            best_state = CognitiveState.CALM
        
        return best_state, confidence
    
    def _generate_insights(self, state: CognitiveState, valence: float, arousal: float, 
                          stress: float, attention: float, modality_results: List[ModalityResult]) -> List[str]:
        """Generate human-readable insights"""
        insights = []
        
        # State-specific insights
        if state == CognitiveState.STRESSED:
            insights.append("You appear to be experiencing elevated stress levels.")
            if stress > 0.8:
                insights.append("Your stress levels are quite high - consider taking a break.")
        elif state == CognitiveState.CALM:
            insights.append("You appear calm and relaxed.")
        elif state == CognitiveState.FOCUSED:
            insights.append("You seem focused and attentive.")
        elif state == CognitiveState.ANXIOUS:
            insights.append("You may be feeling anxious or worried.")
        elif state == CognitiveState.FATIGUED:
            insights.append("You might be experiencing fatigue or low energy.")
        elif state == CognitiveState.EXCITED:
            insights.append("You appear energetic and excited.")
        
        # Metric-specific insights
        if valence < -0.5:
            insights.append("Your emotional state leans negative.")
        elif valence > 0.5:
            insights.append("Your emotional state is quite positive.")
        
        if arousal > 0.8:
            insights.append("Your arousal level is high - you might feel energized or agitated.")
        elif arousal < 0.2:
            insights.append("Your arousal level is low - you might feel calm or tired.")
        
        if attention < 0.4:
            insights.append("Your attention levels appear to be lower than optimal.")
        elif attention > 0.7:
            insights.append("You seem to have good focus and attention.")
        
        # Modality-specific insights
        for result in modality_results:
            if result.confidence < 0.5:
                insights.append(f"The {result.modality} analysis had lower confidence - results may be less accurate.")
        
        return insights[:5]  # Limit to top 5 insights
    
    def _generate_recommendations(self, state: CognitiveState, modality_results: List[ModalityResult]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        if state == CognitiveState.STRESSED:
            recommendations.extend([
                "Try deep breathing exercises to reduce stress",
                "Consider taking a short break or walk",
                "Practice mindfulness or meditation"
            ])
        elif state == CognitiveState.ANXIOUS:
            recommendations.extend([
                "Use grounding techniques (5-4-3-2-1 sensory method)",
                "Practice progressive muscle relaxation",
                "Consider talking to someone about your concerns"
            ])
        elif state == CognitiveState.FATIGUED:
            recommendations.extend([
                "Ensure you're getting adequate sleep",
                "Take regular breaks if working",
                "Consider light physical activity to boost energy"
            ])
        elif state == CognitiveState.EXCITED:
            recommendations.extend([
                "Channel your energy into productive activities",
                "Stay hydrated and mindful of your energy levels"
            ])
        elif state == CognitiveState.CALM:
            recommendations.extend([
                "Maintain this calm state with regular relaxation practices",
                "This is a good time for focused work or reflection"
            ])
        elif state == CognitiveState.FOCUSED:
            recommendations.extend([
                "Take advantage of this focused state for important tasks",
                "Remember to take breaks to maintain this level of focus"
            ])
        
        return recommendations[:3]  # Limit to top 3 recommendations
    
    def _generate_graph(self, valence: float, arousal: float, stress: float, 
                       attention: float, modality_results: List[ModalityResult]) -> Optional[str]:
        """Generate visualization graph as base64 encoded image"""
        try:
            # Create a comprehensive dashboard
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('Cognitive State Analysis Dashboard', fontsize=16, fontweight='bold')
            
            # 1. Valence-Arousal Plot
            ax1.scatter([valence], [arousal], s=200, c='red', alpha=0.7, edgecolors='black')
            ax1.set_xlim(-1, 1)
            ax1.set_ylim(0, 1)
            ax1.set_xlabel('Valence (Negative ← → Positive)')
            ax1.set_ylabel('Arousal (Low ← → High)')
            ax1.set_title('Emotional State')
            ax1.grid(True, alpha=0.3)
            ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
            ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
            
            # Add quadrant labels
            ax1.text(0.5, 0.8, 'Excited', ha='center', va='center', fontweight='bold', alpha=0.7)
            ax1.text(-0.5, 0.8, 'Anxious', ha='center', va='center', fontweight='bold', alpha=0.7)
            ax1.text(0.5, 0.2, 'Calm', ha='center', va='center', fontweight='bold', alpha=0.7)
            ax1.text(-0.5, 0.2, 'Fatigued', ha='center', va='center', fontweight='bold', alpha=0.7)
            
            # 2. Stress and Attention Bars
            metrics = ['Stress Level', 'Attention']
            values = [stress, attention]
            colors = ['#ff6b6b' if stress > 0.6 else '#4ecdc4', '#4ecdc4' if attention > 0.6 else '#ff6b6b']
            
            bars = ax2.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black')
            ax2.set_ylim(0, 1)
            ax2.set_ylabel('Score')
            ax2.set_title('Stress & Attention Levels')
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
            
            # 3. Modality Confidence
            if modality_results:
                modalities = [r.modality.title() for r in modality_results]
                confidences = [r.confidence for r in modality_results]
                
                bars = ax3.bar(modalities, confidences, color='lightblue', alpha=0.7, edgecolor='black')
                ax3.set_ylim(0, 1)
                ax3.set_ylabel('Confidence')
                ax3.set_title('Modality Analysis Confidence')
                ax3.tick_params(axis='x', rotation=45)
                
                # Add value labels
                for bar, conf in zip(bars, confidences):
                    height = bar.get_height()
                    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                            f'{conf:.2f}', ha='center', va='bottom', fontweight='bold')
            
            # 4. Overall Wellness Radar
            categories = ['Valence\n(Positivity)', 'Arousal\n(Energy)', 'Attention\n(Focus)', 'Relaxation\n(Low Stress)']
            values_radar = [
                (valence + 1) / 2,  # Scale valence from [-1,1] to [0,1]
                arousal,
                attention,
                1 - stress  # Relaxation is inverse of stress
            ]
            
            # Close the radar chart
            values_radar += values_radar[:1]
            
            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
            angles += angles[:1]
            
            ax4.plot(angles, values_radar, 'o-', linewidth=2, color='blue', alpha=0.7)
            ax4.fill(angles, values_radar, alpha=0.25, color='blue')
            ax4.set_xticks(angles[:-1])
            ax4.set_xticklabels(categories, fontsize=9)
            ax4.set_ylim(0, 1)
            ax4.set_title('Wellness Profile')
            ax4.grid(True)
            
            plt.tight_layout()
            
            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            image_data = buffer.getvalue()
            buffer.close()
            plt.close()
            
            # Encode as base64
            graph_b64 = base64.b64encode(image_data).decode()
            return f"data:image/png;base64,{graph_b64}"
            
        except Exception as e:
            logger.error(f"Error generating graph: {e}")
            return None
    
    def _default_cognitive_result(self) -> CognitiveStateResult:
        """Return default result when fusion fails"""
        return CognitiveStateResult(
            state=CognitiveState.NEUTRAL,
            confidence=0.5,
            valence=0.0,
            arousal=0.5,
            attention=0.5,
            stress_level=0.5,
            insights=["Analysis could not be completed with confidence"],
            recommendations=["Ensure good data quality for better analysis"],
            modality_contributions={},
            graph_data=None
        )

# Global fusion engine instance
fusion_engine = CognitiveFusionEngine()
