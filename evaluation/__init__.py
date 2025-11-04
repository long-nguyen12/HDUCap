from .bleu import Bleu
from .meteor import Meteor
from .rouge import Rouge
from .cider import Cider
from .tokenizer import PTBTokenizer

def compute_scores(gts, gen):
    metrics = (Bleu(), Meteor(), Rouge(), Cider())
    all_score = {}
    all_scores = {}
    for metric in metrics:
        score, scores = metric.compute_score(gts, gen)
        all_score[str(metric)] = score
        all_scores[str(metric)] = scores

    return all_score, all_scores

def compute_per_sample_bleu(gts, gens, n=4):
    """
    Compute BLEU score for each gts/gen pair.
    
    Args:
        gts: Dictionary with ground truth captions {key: [list of captions]}
        gens: Dictionary with generated captions {key: [caption]} or {key: caption}
        n: Maximum n-gram order for BLEU (default: 4)
    
    Returns:
        Dictionary mapping each key to its BLEU scores:
        {
            key: {
                'BLEU-1': float,
                'BLEU-2': float,
                'BLEU-3': float,
                'BLEU-4': float,
                'BLEU': float,  # BLEU-4 score
                'generated': str,
                'ground_truths': list
            }
        }
    """
    from typing import Union
    
    def normalize_gen(gen):
        """Normalize gen format to list of strings."""
        if isinstance(gen, str):
            return [gen]
        elif isinstance(gen, list):
            return gen
        else:
            raise ValueError(f"Unexpected gen format: {type(gen)}")
    
    # Normalize gens format
    normalized_gens = {}
    for key in gens:
        normalized_gens[key] = normalize_gen(gens[key])
    
    # Ensure all keys match
    if set(gts.keys()) != set(normalized_gens.keys()):
        # Use intersection of keys
        common_keys = set(gts.keys()) & set(normalized_gens.keys())
        gts = {k: gts[k] for k in common_keys}
        normalized_gens = {k: normalized_gens[k] for k in common_keys}
    
    if not gts:
        return {}
    
    # Compute BLEU scores
    bleu_metric = Bleu(n=n)
    aggregate_scores, per_image_scores = bleu_metric.compute_score(gts, normalized_gens)
    
    # per_image_scores is a list of lists: [[B1_0, B1_1, ...], [B2_0, B2_1, ...], ...]
    # Extract BLEU scores for each sample
    per_sample_results = {}
    sorted_keys = sorted(gts.keys())
    
    if not per_image_scores or not per_image_scores[0]:
        return {}
    
    for i, key in enumerate(sorted_keys):
        if i < len(per_image_scores[0]):  # Check if we have scores for this index
            # Get BLEU scores for this sample (all n-gram orders)
            bleu_scores = {
                f'BLEU-{k+1}': per_image_scores[k][i] if k < len(per_image_scores) and i < len(per_image_scores[k]) else 0.0
                for k in range(n)
            }
            # BLEU-4 is typically the main score
            bleu_scores['BLEU'] = bleu_scores.get(f'BLEU-{n}', 0.0)
            
            per_sample_results[key] = {
                'bleu_scores': bleu_scores,
                'generated': normalized_gens[key][0] if normalized_gens[key] else '',
                'ground_truths': gts[key] if isinstance(gts[key], list) else [gts[key]]
            }
    
    return per_sample_results