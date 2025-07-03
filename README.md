# Improving Turbulent Radiative Layer Predictions Using Computer Vision

Master Information Studies: Data Science 
Faculty of Science
University of Amsterdam

**Author**: Kato Mari Roos Schmidt  
**Email**: kato.schmidt@student.uva.nl  
**Supervisor**: Ana Lucic (UvA) – a.lucic@uva.nl

## Abstract

Turbulent flows are notoriously difficult to predict due to their chaotic nature and high computational demands. In astrophysics, Turbulent Radiative Layers (TRLs) represent a particularly complex case, where radiative cooling and turbulent mixing interact dynamically. While recent research has applied deep learning models to such systems, it remains unclear how architectural choices affect performance, especially under long autoregressive rollouts.

Existing benchmarks, such as CNextU-Net, set a strong baseline but leave room for exploration. This study evaluates three compact and purely data-driven models: CNextU-Net, SineNet, and SwinNet. All models are assessed on the TRL-2D dataset under equal computational constraints, using the Variance-Relative Mean Squared Error (VRMSE) as the evaluation metric.

Results show that SineNet variants outperform the benchmark in short-term prediction, while SwinNet offers more stable performance in long rollouts. However, none of the models achieve consistent gains across all tasks. A persistent challenge is the accurate prediction of the pressure field, which dominates error accumulation. These findings clarify the trade-offs between architectural design, model performance, and computational efficiency in turbulent systems, and highlight refinement-based approaches or those combining the strengths of SineNet and SwinNet as a promising path for future improvement.

## Keywords

Turbulent Radiative Layer, Computer Vision, Deep Learning, Autoregressive Rollout
