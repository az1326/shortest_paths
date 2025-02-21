# This file contains parameters specifying the configurations of graphs

########### BINOMIAL GRAPHS ###########
BINOM_P_DICT = {
    20000: 0.0005,
    40000: 0.00025,
    100000: 0.0001
}
BINOM_LO = 3
BINOM_HI = 18

########### POWER LAW GRAPHS ###########
# rand mean = 11.38, snow mean = 14.30
POWER_A_GAMMA = -2
POWER_A_LO = 6
POWER_A_HI = 30

# rand mean = 8.62, snow mean = 9.72
POWER_B_GAMMA = -3
POWER_B_LO = 6
POWER_B_HI = 20


########### SBM GRAPHS ###########
SBM_BLOCK_SIZE = 172
SBM_NUM_BLOCKS_DICT = {
    20000: 130,
    40000: 260,
    100000: 650
}
SBM_N_DICT = {
    20000: SBM_BLOCK_SIZE * SBM_NUM_BLOCKS_DICT[20000],
    40000: SBM_BLOCK_SIZE * SBM_NUM_BLOCKS_DICT[40000],
    100000: SBM_BLOCK_SIZE * SBM_NUM_BLOCKS_DICT[100000]
}
SBM_P_WITHIN = 0.0832308110765883
SBM_P_ACROSS = 4.7181783381111744e-05

SBM_WITHIN_LO = 5
SBM_WITHIN_HI = 25
SBM_ACROSS_LO = 0
SBM_ACROSS_HI = 5

########### GENERAL ###########
MAX_DIST = 7
