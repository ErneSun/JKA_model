# Technology adoption declaration

| technology | decision | reason |
|---|---|---|
| frozen online residual target | ADOPT | identifies the error of the actual inference backbone |
| post-Koopman direct Δz | ADOPT | exact match to the V0.7 question |
| fixed-history tiny MLP | ADOPT | smallest non-Markovian capacity |
| parameter-matched instantaneous control | ADOPT | separates capacity from history |
| shuffled-history control | ADOPT | tests temporal ordering |
| multi-H sweep and 95% effective horizon | ADOPT | distinguishes Markovian, finite, and unresolved long memory |
| ACF-only memory decision | REJECT | correlation does not establish useful closed-loop memory |
| Neural DDE | DEFER | new solver/delay contract |
| RNN/GRU/LSTM | DEFER | hidden-state complexity before memory evidence |
| Mamba | DEFER | recent preprint; unnecessary for short fixed history |
| attention-free transformer | DEFER | different correction order and re-encoding |
| joint V0.6 fine-tuning | REJECT | destroys causal attribution |
| exact MZ-kernel claim | REJECT | not mathematically identified |

This declaration is binding for V0.7 implementation and GPU validation.
