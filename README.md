# MBPCA-SHM

# Preface

The uncertainty of environmental hazards attributed to climate change poses an imminent
threat to the longevity and lifespan of critical infrastructure components. Structural Health
Monitoring (SHM) is the process by which a continuous, real-time mechanism is employed for
use in damage identification of structural components. SHM often involves the instrumentation
of structural elements with a dense, and distributed network of sensors from which the condition
of such elements may be monitored in real-time. Other monitoring schemes may necessitate an
analytical approach and are facilitated in combination with finite element models. However,
existing techniques prescribed through SHM may lack the capability to cope with anomalous
changes in the conditions under which critical infrastructure components are monitored.

Vibration-based structural health monitoring enables the observation of the global
frequency response for a given structure for the purposes of damage identification and
classification. Vibration-based SHM leverages real-time monitoring through instrumentation
of structural components with sensors (i.e., MEMS accelerometers, Fibre-Bragg Gratings,
piezoelectric sensors) in the assessment of infrastructure health through analysis of the
frequency response of such components. The non-destructive retrofitting of civil infrastructure
and structural components with a distributed network of sensors enables a data-driven approach
to modal analysis and is an effective means for which structural deficiencies may be identified.
The case study described herein explores a data-driven approach to vibration-based SHM for
damage identification of the KW51 railway bridge in Leuven, Belgium.

Due to the volume of data retained for analysis, it is important to note the limitations
intrinsic to existing implementations of multi-block PCA. Specifically, existing python-based
implementations included in the trendfitter and scikit-learn libraries are not well-equipped for
big data and multi-block applications respectively. These limitations form the motivation for
the second key objective in this study. That is to say, the capacity of existing python-based
implementations for handling big data is limited, thus, a novel approach shall be taken to
overcome these limitations. 

In the original study carried-out by Maes et. Lombaert, they employed 
a robust PCA-based approach. The study conducted in this repository employs a consensus (multi-block) PCA approach. The dataset has been made public by the original authors of the study at:
https://zenodo.org/records/3745914.

