# partial_discharge_detection

Code used for competition:

https://www.kaggle.com/c/vsb-power-line-fault-detection

Best place was 8th a few weeks into the comp. using signal conversion to spectrogram, ResNet34 and simple image flip 
augmentation.

This basic approach was continued throughout, lstm was trialed with poor results.

Significant time spent on data augmentation - eg removing LF component, recreating HF componet then recombining,
and using deeper networks but couldn't keep place in rankings.
