# Celeris - [Demo](https://wrong-wallis-celeris-bbabebfc.koyeb.app/)

A fast chess engine with nnue trained by TheGogy

## Huge thanks to TheGogy for contributing to this project!

### CCRL Blitz: 3130 Elo

### Move generation
* Magic Bitboards ([wiki](https://www.chessprogramming.org/Magic_Bitboards))
  * PEXT intrinsics (Optional) ([wiki](https://www.chessprogramming.org/BMI2#PEXTBitboards))
  * Fancy Magic Bitboards ([wiki](https://www.chessprogramming.org/Magic_Bitboards))
* Fully Legal and Fast Move Generator
  * Keep tracks of pins and checks etc
  * Fast move generation (Up to 600 Mnps on a R9 7900)
### Move ordering
* Killer Move Heuristics
* History Table ([wiki](https://www.chessprogramming.org/History_Heuristic))
* MVV-LVA ([wiki](https://www.chessprogramming.org/MVV-LVA))
* Staged Move Generation
* Static Exchange Evaluation ([wiki](https://www.chessprogramming.org/Static_Exchange_Evaluation))
### Search
* Iterative Deepening ([wiki](https://www.chessprogramming.org/Iterative_Deepening))
* Classic Alpha Beta Search ([wiki](https://www.chessprogramming.org/Alpha-Beta))
  * Negamax
* Quiescence Search ([wiki](https://www.chessprogramming.org/Quiescence_Search))
* Transposition Table ([wiki](https://www.chessprogramming.org/Transposition_Table))
  * Lockless Hashing
* Null Move Pruning ([wiki](https://www.chessprogramming.org/Null_Move_Pruning))
* Late Move Pruning 
* Principal variation search ([wiki](https://www.chessprogramming.org/Principal_Variation_Search))
### Evaluation
* NNUE evaluation (Train by TheGogy)

### Demo link on github main page
