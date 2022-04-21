"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Referring Expression dataset
"""
import random
import numpy as np
import json
import torch
from torch.nn.utils.rnn import pad_sequence
from toolz.sandbox import unzip
from .data import (DetectFeatTxtTokDataset, TxtTokLmdb, DetectFeatLmdb,
                   TxtLmdb, pad_tensors, get_gather_index)

fname_dict = {}
sample_dict = {'reverie_JeFG25nYj2p_d907086855c9497dacdb77503099ba10_00.npz_317':{},
'reverie_VLzqgDo317F_259fd84d195d4d9bac4e14bdf953521d_01.npz_973':{},
'reverie_VzqfbhrpDEA_ed1015d3d3f84c79885ded306617eeb4_16.npz_263':{},
'reverie_S9hNv5qa7GM_8b219cad487e4091b7c29146f0224729_05.npz_1775':{},
'reverie_D7N2EKCX4Sj_cd0016747a75487f9fe60f9934d67af5_00.npz_2002':{},
'reverie_p5wJjkQkbXX_1ac9330105c84fe3bf9058aebfd26f6a_03.npz_2205':{},
'reverie_82sE5b5pLXE_f3aa7887b82f4e1787dca604277580bd_00.npz_2422':{},
'reverie_p5wJjkQkbXX_a819af38ac3f4e54b7cdc834c04a1d05_00.npz_2733':{},
'reverie_ac26ZMwG7aT_cc03fd369f254cd985894b77fd8babde_08.npz_3256':{},
'reverie_29hnd4uzFmX_15b25adcf9bb4a0f8c751138a651dd8b_00.npz_1731':{},
'reverie_759xd9YjKW5_8aab1e4ed6e845328b9d8ab7e2c009ac_01.npz_3857':{},
'reverie_PX4nDJXEHrG_f5933619d2914bdca7a58a004ec7c3df_01.npz_3916':{},
'reverie_5q7pvUzZiYa_31447d5126954b3aba54bebd9f768964_04.npz_4462':{},
'reverie_8WUmhLawc2A_48c8e8de5830459aa06ebaa4ae2b91ad_02.npz_4519':{},
'reverie_p5wJjkQkbXX_17535b07dd374560a191b4e0d092b479_07.npz_4854':{},
'reverie_JeFG25nYj2p_d3864482f6bf42ada7aa0574b0f69f48_00.npz_5060':{},
'reverie_ur6pFq6Qu1A_aaa0537ff1f4428092ae5c195200d5ca_03.npz_5375':{},
'reverie_ULsKaCPVFJR_c73a6f3769bc471588158d99186b80e6_07.npz_5523':{},
'reverie_p5wJjkQkbXX_eec88eb9fe534ed79ef172a3e115f54a_06.npz_5763':{},
'reverie_r1Q1Z4BcV1o_d08536af13b44cf486b3d906d76cfaf8_07.npz_6008':{},
'reverie_ac26ZMwG7aT_d1ffe5280fce4ac5a949cdc9ee8b6f7c_04.npz_6480':{},
'reverie_e9zR4mvMWw7_d2875e02333d4dd991e866786a87c1be_18.npz_7100':{},
'reverie_7y3sRwLe3Va_f6a03fba4dfd415a8b65cce84ee21c19_00.npz_7331':{},
'reverie_mJXqzFtmKg4_7478c47d6145458d80f89cee31d70fdf_01.npz_7421':{},
'reverie_mJXqzFtmKg4_b18be3157a6f4b9c8df154f8f310099d_08.npz_5039':{},
'reverie_rPc6DW4iMge_dee9919470d2404087727c29e18de0e3_10.npz_8323':{},
'reverie_759xd9YjKW5_3d0ecd8a8a6f4ba2b3849f2b77083438_00.npz_8640':{},
'reverie_EDJbREhghzL_23c4fc0e00a741c8a631f859af1af23c_03.npz_1839':{},
'reverie_e9zR4mvMWw7_86c7e095a5bd46cf8d2e286ab67d7ded_03.npz_8711':{},
'reverie_7y3sRwLe3Va_1838a8fb8e9f48cebcd079d5373cba02_26.npz_8795':{},
'reverie_D7N2EKCX4Sj_f8190b68f53e40268771352a91bab873_00.npz_7762':{},
'reverie_uNb9QFRL6hY_978d7a8eb0794936bd8fd092306e1dc5_03.npz_9368':{},
'reverie_V2XKFyX4ASd_50e5eb51e92446a78ccfb851123e801d_06.npz_9447':{},
'reverie_8WUmhLawc2A_c6221b7f72c04623a8396e3f3bb14bfa_15.npz_10782':{},
'reverie_jh4fc5c5qoQ_c91341f14a4f499e8b45d128091ed087_26.npz_10937':{},
'reverie_5LpN3gDmAk7_0d31b7f9341e40c9ab508eb53168dfde_00.npz_11098':{},
'reverie_29hnd4uzFmX_4d574a196c884caa93b2da69dd36f201_14.npz_11338':{},
'reverie_JmbYfDe2QKZ_1316450be64b472c9f84f6f71a991757_07.npz_4089':{},
'reverie_ur6pFq6Qu1A_efd08e9fd6e84cfcb3aee4af348a0aa8_04.npz_11749':{},
'reverie_VVfe2KiqLaN_88fda52a9c9e4a5cb1774e741ecce8f6_00.npz_11948':{},
'reverie_7y3sRwLe3Va_554ecce763be4bd49e89c1bfb89676f3_08.npz_8433':{},
'reverie_cV4RVeZvu5T_5bc41a6e3b7748149e0e8592c5b4d142_01.npz_12525':{},
'reverie_E9uDoFAP3SH_099839b0911a409787e237301c90e418_00.npz_12583':{},
'reverie_7y3sRwLe3Va_a775c7668ca9419daaf506e76851821e_06.npz_3818':{},
'reverie_D7N2EKCX4Sj_c69af2f0e57b485886e0f4667b2209a5_00.npz_4366':{},
'reverie_D7N2EKCX4Sj_f67fc5e261794b2b8c4ef494604d2059_06.npz_383':{},
'reverie_D7N2EKCX4Sj_9766d89c21094c5b872be3a378d4cbad_04.npz_13404':{},
'reverie_EDJbREhghzL_feb3e37459f942f98d53eb78d66375f5_01.npz_14661':{},
'reverie_cV4RVeZvu5T_b2af08ee382d4746b235a45d0a7032bb_08.npz_14915':{},
'reverie_i5noydFURQK_533a4cd1505a486fa8801bb3aa38f2e3_07.npz_9503':{},
'reverie_mJXqzFtmKg4_972a6c55f1fd40ff8f32fab7fe82f5e7_09.npz_15190':{},
'reverie_82sE5b5pLXE_dc524a4004ae4faaaf8e4978f7ad7897_01.npz_15302':{},
'reverie_uNb9QFRL6hY_9ec5ebcf3edc44e1ba6dd02e8423e5ed_01.npz_15682':{},
'reverie_ac26ZMwG7aT_10b5aaa0dd094b00a74dc52823b7aa66_09.npz_15966':{},
'reverie_VFuaQ6m2Qom_8a4ee4cfaba847cc8e68500039c9a916_07.npz_16307':{},
'reverie_Vvot9Ly1tCj_8c6c10daaa394125b4bdb198bc99cbad_05.npz_4006':{},
'reverie_uNb9QFRL6hY_ce21aa1c088c431f937dcba171c9961d_02.npz_17059':{},
'reverie_ac26ZMwG7aT_7b14552fe83b4e76974f711b8fcfc46f_06.npz_16158':{},
'reverie_8WUmhLawc2A_c75fd37598a0457e83b654dbd0990934_20.npz_17533':{},
'reverie_1pXnuDYAj8r_7cdc2565c773401289b95360da22e017_07.npz_5607':{},
'reverie_V2XKFyX4ASd_cae5bc03a39a4790aa93f8ce3a55da06_04.npz_17784':{},
'reverie_ac26ZMwG7aT_11fae969386e4e4fa6ed73ac9f20cd1d_04.npz_17941':{},
'reverie_B6ByNegPMKs_73f178674ed0409ba292fbc4d057a1ec_27.npz_18507':{},
'reverie_759xd9YjKW5_fc30206582bb494190095c4439e2da29_04.npz_9589':{},
'reverie_jh4fc5c5qoQ_0565dcc2a0d54d72b97f26a1a14cebe5_07.npz_18793':{},
'reverie_ac26ZMwG7aT_c8e6c588d25d42cd8400a07db98c40fe_02.npz_18830':{},
'reverie_1LXtFkjw3qL_95e9e714db234174b904782f1256c3f9_08.npz_14455':{},
'reverie_759xd9YjKW5_d7457234e5d54417bcc64a74d6ecfd9a_01.npz_19128':{},
'reverie_r47D5H71a5s_869e52d579cc4c9a85979d3e20eb2455_00.npz_19573':{},
'reverie_EDJbREhghzL_553465b804fb4fc29bf530f71dfdf1d5_00.npz_5656':{},
'reverie_Vvot9Ly1tCj_15ef51c638de46f196fdd00a805e80c4_06.npz_19797':{},
'reverie_E9uDoFAP3SH_21d98b3051984309af072f3fde91c49d_01.npz_20207':{},
'reverie_5LpN3gDmAk7_fc8991e507984600b21065aef5b410b5_00.npz_20351':{},
'reverie_B6ByNegPMKs_5f316f408d4644baa49d6a64a123bb88_12.npz_20590':{},
'reverie_759xd9YjKW5_1015e20d2dca4a01b6cca0ca0d71c29a_07.npz_21100':{},
'reverie_1pXnuDYAj8r_deb99671679648e28ccf7d622051ab46_08.npz_3408':{},
'reverie_JF19kD82Mey_c20f944e9c434804944b0c1251247202_00.npz_8339':{},
'reverie_8WUmhLawc2A_52be196a34fd415fa984b16aa270481c_04.npz_6843':{},
'reverie_VzqfbhrpDEA_dfb1f5674ff1434f8e101bcd9d3cfb20_00.npz_21540':{},
'reverie_1LXtFkjw3qL_40eda297975f4648bbc301ff98940fcf_02.npz_21837':{},
'reverie_r1Q1Z4BcV1o_31dfaa2e67354657a321602ab74e9b02_00.npz_21982':{},
'reverie_B6ByNegPMKs_4f85ccd0b873491483317feda781d699_18.npz_22161':{},
'reverie_1LXtFkjw3qL_e5c5d8fa1ad94d97bcdb74e69e938ea0_04.npz_22323':{},
'reverie_dhjEzFoUFzH_9ef6399314c94a93abeec2d6054f4658_00.npz_23046':{},
'reverie_vyrNrziPKCB_13539678a339494dbe249d3e8137778d_03.npz_23262':{},
'reverie_SN83YJsR3w2_09be981423a3442998bd724cbbdf9b2b_04.npz_13175':{},
'reverie_vyrNrziPKCB_3193cd5f6d7a4b2bbf85d5043cbc6352_09.npz_17362':{},
'reverie_D7N2EKCX4Sj_5c10e79b2673419ebf8d0f84582961ab_05.npz_5194':{},
'reverie_8WUmhLawc2A_e0a8257d73b84c7eb0f227c52db0e51c_08.npz_24345':{},
'reverie_cV4RVeZvu5T_3393cfe75d8e4593bf4b05d00b7eb2c2_00.npz_8014':{},
'reverie_HxpKQynjfin_5cd96a7c879547eda2a90077c79bc7f0_03.npz_19813':{},
'reverie_JmbYfDe2QKZ_267636f4842a422ba13275cb995ca9f5_05.npz_24729':{},
'reverie_B6ByNegPMKs_8e3cb86643cc4eee947faae80d8fb256_08.npz_24740':{},
'reverie_SN83YJsR3w2_35298dd1214345239f966099a1ddac3f_00.npz_24926':{},
'reverie_EDJbREhghzL_df428e69750340cdb9612ca81d9b098b_06.npz_25679':{},
'reverie_p5wJjkQkbXX_4e2df346f67d4412b59df955aa0650c8_07.npz_25783':{},
'reverie_8WUmhLawc2A_1b48df86b7a149fa8e90161265def866_06.npz_25873':{},
'reverie_JF19kD82Mey_c7b357a72c3d4ce2aa16f8d5c41a54ba_06.npz_26090':{},
'reverie_S9hNv5qa7GM_90ec6d25cc584b50ad6c56ec683eeb6b_08.npz_26231':{},
'reverie_e9zR4mvMWw7_18653fa3d6ba4f82889237201ee07d11_02.npz_26328':{},
'reverie_e9zR4mvMWw7_1daae4b7becc43949516096170ce2a76_06.npz_26863':{},
'reverie_759xd9YjKW5_bbbe225834074bb0a3fb31a3e1e80685_00.npz_27185':{},
'reverie_qoiz87JEwZ2_1e569e030be24ccf8e8f6ccb42007846_08.npz_27328':{},
'reverie_HxpKQynjfin_96478d4e66104792b0481887c85255a8_06.npz_27422':{},
'reverie_1LXtFkjw3qL_541ab4e8ff1f4bc8bf20775a1fb38872_00.npz_27518':{},
'reverie_JF19kD82Mey_f1b191033043441987b8ebf1bb55002c_12.npz_27639':{},
'reverie_ZMojNkEp431_6855b783e2f34c7981b90ddb5de81bd7_05.npz_27681':{},
'reverie_jh4fc5c5qoQ_39791f422b0244aa84b3d41d9b218f7e_02.npz_27778':{},
'reverie_JeFG25nYj2p_2fb3e5986a7b45ceb6d83f975a5ae6c3_05.npz_28028':{},
'reverie_JmbYfDe2QKZ_38a021bef7634782b42c0583f84c6b8e_00.npz_28127':{},
'reverie_B6ByNegPMKs_fa6d662db40c4335b899082f62b51111_13.npz_28228':{},
'reverie_EDJbREhghzL_66d32e6e1f5847bab89af0d5c61c120d_04.npz_28318':{},
'reverie_rPc6DW4iMge_01b11b701b5240a48be2a86171daa245_00.npz_28061':{},
'reverie_D7N2EKCX4Sj_cc7f149e59eb47fd8bde336f2bf3391d_17.npz_28488':{},
'reverie_D7N2EKCX4Sj_9e94e381ffbf44eabc4b3fb0f8bff78a_05.npz_29004':{},
'reverie_7y3sRwLe3Va_f0db0d83f9d646b997d099d7eae80dd1_07.npz_29120':{},
'reverie_29hnd4uzFmX_09f015ea8ca94df8841bcea7030bca99_12.npz_29454':{},
'reverie_8WUmhLawc2A_4304b16fd7b744ec8cd9277d0adcb4da_06.npz_29659':{},
'reverie_B6ByNegPMKs_b3401d49d1364e87b067e354154ba4bd_21.npz_29780':{},
'reverie_qoiz87JEwZ2_aaa8773c86214e9990d4c3ede3223499_28.npz_29928':{},
'reverie_JmbYfDe2QKZ_e58daa54fe36484984da0fc3e209ee39_00.npz_26613':{},
'reverie_ac26ZMwG7aT_58f71a450eed4d878a9b9e6a67ad6f39_24.npz_30145':{},
'reverie_ULsKaCPVFJR_28a37b7b94a34889ab5a774f43219fc3_11.npz_30271':{},
'reverie_aayBHfsNo7d_d1af2f90794243db807d60a07050941c_01.npz_30280':{},
'reverie_8WUmhLawc2A_a59713eed44e47cca397462601c4d960_02.npz_30676':{},
'reverie_VzqfbhrpDEA_df42352940c8404e955eafca30659120_14.npz_30705':{},
'reverie_1pXnuDYAj8r_c3b8b8e1f0994859869b42fb760fce9f_04.npz_31016':{},
'reverie_2n8kARJN3HM_ffacc18a27644508a6e4223823a0e2a4_02.npz_31124':{},
'reverie_i5noydFURQK_21e7d627b7b34355a52f3c88cb2ad446_03.npz_31333':{},
'reverie_b8cTxDM8gDG_357424fc423a4e978fa88a8e4d3bed78_12.npz_31389':{},
'reverie_JmbYfDe2QKZ_ac32217f29704340bf467f51b7f15786_29.npz_31522':{},
'reverie_ULsKaCPVFJR_a085b1aee4be45b6982a1f7b601654eb_08.npz_19789':{},
'reverie_D7N2EKCX4Sj_7f83a9b5299744cc8bc32be5d94108c0_12.npz_31893':{},
'reverie_PX4nDJXEHrG_23d3a1b1dc134acc8ffa787e2ee558ed_07.npz_31962':{},
'reverie_Vvot9Ly1tCj_624b17d75c344a0e8122c7b0c6858dce_13.npz_32409':{},
'reverie_ac26ZMwG7aT_0bd07b7213b245f8a54ec4010f6ef1cc_00.npz_32512':{},
'reverie_cV4RVeZvu5T_6333e669c4174c5dbcbc70e6da1c08f6_07.npz_19244':{},
'reverie_PX4nDJXEHrG_8646134295d2446ba7e27db2fd6da710_12.npz_32605':{},
'reverie_1LXtFkjw3qL_7d01dee740ef4288b111dc65f449ca22_08.npz_32730':{},
'reverie_82sE5b5pLXE_89eeae92b2fe4b68abb274970a931608_02.npz_26961':{},
'reverie_8WUmhLawc2A_92ffe9fa9ea6437c88eac23b0bafc0bc_05.npz_30229':{},
'reverie_B6ByNegPMKs_bf4ff06d86464dcb83b52ac92bdcafea_24.npz_32927':{},
'reverie_759xd9YjKW5_f5f240dba39b4c19a1354c89c56d7a59_02.npz_32973':{},
'reverie_qoiz87JEwZ2_7f21726e1efe494592c3278ce33cd622_00.npz_33029':{},
'reverie_ac26ZMwG7aT_3317348467784b19925a007204cb7f17_07.npz_33098':{},
'reverie_jh4fc5c5qoQ_b000c5baa76b454caa1c58c9aac585f6_00.npz_33307':{},
'reverie_qoiz87JEwZ2_223b4ced68cd4688b6612a2795409490_13.npz_33427':{},
'reverie_mJXqzFtmKg4_b419f6ecc3f24120afbb65d25efcb444_00.npz_33735':{},
'reverie_ac26ZMwG7aT_e394a9eecb63432b804ebb2e96a563b5_00.npz_33906':{},
'reverie_7y3sRwLe3Va_93292ae9057244519e582c2c53abfb3e_06.npz_14756':{},
'reverie_EDJbREhghzL_181111f8575a49039444d6180bf71c10_05.npz_34058':{},
'reverie_rPc6DW4iMge_e0eaabf84a714ad4ba5000604fa11b2b_04.npz_34343':{},
'reverie_pRbA3pwrgk9_4fe4474e9bf9429ea92d202bdea4dce5_03.npz_34593':{},
'reverie_ac26ZMwG7aT_067ec17ab4314effbaf5e67e5acfacad_01.npz_34854':{},
'reverie_ac26ZMwG7aT_6574e941f0be49afa9fd447b99b2e783_00.npz_31698':{},
'reverie_1pXnuDYAj8r_efa99d80f5024c999633851f5875c6c7_07.npz_35196':{},
'reverie_sT4fr6TAbpF_c32e11fd03ac4b2bb88aafacff21527f_01.npz_35617':{},
'reverie_i5noydFURQK_0e45c1af5098415c9bbd3a6ce537c7cb_14.npz_30557':{},
'reverie_SN83YJsR3w2_b227c3cafc50455eaa601243127e26ec_00.npz_35732':{},
'reverie_p5wJjkQkbXX_833c62d173f14aaabd2618c4aa1aefc2_05.npz_35923':{},
'reverie_1pXnuDYAj8r_c3c0a1a738a749a7a986d64c872b7671_04.npz_12829':{},
'reverie_VzqfbhrpDEA_f90f7e3981ac4510906239ff3422c9e8_21.npz_36151':{},
'reverie_rPc6DW4iMge_f378f8971f7b41ccb0f2c1dc14ba290d_06.npz_1163':{},
'reverie_cV4RVeZvu5T_7cd02069ac1546319b95be27fc04d7b5_00.npz_36388':{},
'reverie_VLzqgDo317F_4335216422a24ad9bd6a662531d49f90_12.npz_36547':{},
'reverie_82sE5b5pLXE_acee0721d5ec4a80a29f2a2cbadc4bad_00.npz_36986':{},
'reverie_aayBHfsNo7d_5278ace992664bbcb69d686a7be2c3b3_02.npz_37058':{},
'reverie_vyrNrziPKCB_6fe29c72b80b449492666e34067f9008_26.npz_37302':{},
'reverie_jh4fc5c5qoQ_8554983cf79243f6bfb0f6fa8950e8a5_14.npz_37380':{},
'reverie_sKLMLpTHeUy_dfa0373deb9d4e5db88b76c95dc0d6a9_00.npz_37410':{},
'reverie_XcA2TqTSSAj_3510eb416b7d4d779881abc5b5cf2cec_00.npz_37456':{},
'reverie_gTV8FGcVJC9_ec50addd94df4040b407bf46861cf5e3_06.npz_37561':{},
'reverie_JmbYfDe2QKZ_c69e398f10eb40938d19c2c736b0c084_07.npz_38179':{},
'reverie_1pXnuDYAj8r_6a3fba49106e4a9fb40cd0bf47dd4d46_00.npz_28352':{},
'reverie_vyrNrziPKCB_c929b31b97a14f91b1cadb318a4a8353_00.npz_38279':{},
'reverie_VFuaQ6m2Qom_8f0adb0cce834e04a0bf92f48219c04d_04.npz_38339':{},
'reverie_D7N2EKCX4Sj_7f39d79ddfec4b6e9c98e51576d59ab1_12.npz_38448':{},
'reverie_gTV8FGcVJC9_c7b8d353fead4ab2a121873553abe024_01.npz_38743':{},
'reverie_82sE5b5pLXE_c22f5ecc360048698e71cad6542f07a6_07.npz_6753':{},
'reverie_8WUmhLawc2A_b6018ae2d13248f7a12edbf31f70b04d_04.npz_38797':{},
'reverie_qoiz87JEwZ2_6a15f715d61843b2a84845d392ec0b72_06.npz_13833':{},
'reverie_7y3sRwLe3Va_3d21c0cbc98b4c3da6585fabef9c68c3_03.npz_39358':{},
'reverie_pRbA3pwrgk9_c28d78b8e9c94e89aac08c1f75804c3d_02.npz_40027':{},
'reverie_mJXqzFtmKg4_303468f25f254a04ba109f567acb4340_28.npz_40065':{},
'reverie_E9uDoFAP3SH_bffa1faac5024b19abf8b4150e634c9c_07.npz_40259':{},
'reverie_GdvgFV5R1Z5_97ed68de989e44fdaf2d9b949898fab6_00.npz_40331':{},
'reverie_mJXqzFtmKg4_e428c761024a458e8b052494cf6249f5_19.npz_40497':{},
'reverie_sT4fr6TAbpF_cc1777c97c9e473dab1d046e56e1961a_00.npz_40530':{},
'reverie_sT4fr6TAbpF_2f5285734159491fa089541bf820e528_00.npz_40585':{},
'reverie_sT4fr6TAbpF_5d8d5fbcc95e47b4be3ef51862b4d1f8_00.npz_40667':{},
'reverie_1pXnuDYAj8r_b12981cba2ba4525b377ef503d92b843_06.npz_40745':{},
'reverie_i5noydFURQK_3560fdb7b97c462ab565c8946b77ecef_29.npz_41233':{},
'reverie_uNb9QFRL6hY_75e2851e5734470a813afe0afdcebb62_03.npz_41337':{},
'reverie_ZMojNkEp431_6944cb3d349a424899b2ed5b4c972763_07.npz_41504':{},
'reverie_ZMojNkEp431_12eaa3a3a39e4c9fa5106812cb7da084_01.npz_42716':{},
'reverie_Uxmj2M2itWa_5172278505b14dbfba2e900d89ec3d3d_00.npz_42978':{},
'reverie_pRbA3pwrgk9_3851376d4f84494ebdc080b34ddc0f5e_04.npz_34963':{},
'reverie_vyrNrziPKCB_de4d37507ef94c05b71156cdb6af3067_08.npz_44049':{},
'reverie_EDJbREhghzL_e3b5cbc0554e476a8f17d4797b993d7a_12.npz_44552':{},
'reverie_2n8kARJN3HM_4903a8de1d0945b6bd815f11aa2850e9_07.npz_44815':{},
'reverie_r1Q1Z4BcV1o_77cfb227d30443428d4c7787a3da713c_14.npz_45011':{},
'reverie_ZMojNkEp431_cff2c62e83724edda036bbe481bf181b_00.npz_45349':{},
'reverie_qoiz87JEwZ2_b411496c1f7b403981f0c2f24995309c_05.npz_43833':{},
'reverie_GdvgFV5R1Z5_6178647ca8d14dc09370f6c1b7ed2fd6_08.npz_45815':{},
'reverie_aayBHfsNo7d_0608309a14594ef9ac53c26b9e03a1ae_00.npz_46195':{},
'reverie_8WUmhLawc2A_276586d2d2e14dde82b112c03ccf2188_06.npz_46454':{},
'reverie_ZMojNkEp431_9c90566989cb4f9f80b711d1a9876e1a_02.npz_46799':{},
'reverie_ZMojNkEp431_cc7528acf922480cb0a9169d4e539e70_07.npz_46942':{},
'reverie_JmbYfDe2QKZ_b03eb3aa3b6e4c5abb5e78830372ff89_05.npz_28623':{},
'reverie_Vvot9Ly1tCj_865e21984c724badb6eba2652a79d596_00.npz_47374':{},
'reverie_SN83YJsR3w2_ef1bed677e2545629b7b68cf4d181d18_00.npz_47488':{},
'reverie_ac26ZMwG7aT_982920829a0b433880410222539f240e_03.npz_48405':{},
'reverie_ac26ZMwG7aT_f4939bf6f00a4864832a358f1ea8394e_08.npz_41128':{},
'reverie_82sE5b5pLXE_bbb58ec45ea54a879252ecefa351559b_08.npz_48431':{},
'reverie_mJXqzFtmKg4_05c5f3286f8a433ba542fa13ab0a8c7b_09.npz_49179':{},
'reverie_vyrNrziPKCB_53b98ee4dfc1481ab5750f44ebdee37c_04.npz_49608':{},
'reverie_S9hNv5qa7GM_2c449dc925de472ca31124feca25a01a_00.npz_49986':{},
'reverie_5q7pvUzZiYa_d3b6b42577fc4c9ea6bb46a8cc9f729c_07.npz_50311':{},
'reverie_qoiz87JEwZ2_3772d953cd264ebb9ff730bde0bf842c_01.npz_51530':{},
'reverie_7y3sRwLe3Va_99b1210b63c94f9184a9f06032a2ea4a_32.npz_51635':{},
'reverie_vyrNrziPKCB_f83d0032ce684933a4d3e8784a9fd0bf_01.npz_51737':{},
'reverie_5q7pvUzZiYa_2f78f0c5a7714e29a93fe5cb72bea6bc_09.npz_51987':{},
'reverie_r1Q1Z4BcV1o_06c88433ed80477f90bd50f8dc25e7de_00.npz_52556':{},
'reverie_ZMojNkEp431_1f701e524a6d48abbfdf40cd288cf1dd_00.npz_52737':{},
'reverie_ac26ZMwG7aT_4bd7d1a0c683471babf8c7abab4121a3_16.npz_43206':{},
'reverie_EDJbREhghzL_b897a3508d9f43e19ba2b8d48323445a_01.npz_31636':{},
'reverie_B6ByNegPMKs_4f232bec0d894e798b504ccb73b0eed8_04.npz_54350':{},
'reverie_qoiz87JEwZ2_3f90fa45a13a4a8e8b071f30f5ade3be_04.npz_54446':{},
'reverie_JmbYfDe2QKZ_9f7f95239ff04f7581f1cf9a4b5a9eaf_03.npz_54699':{},
'reverie_D7N2EKCX4Sj_be8ffeba688d499b982fa36b58f0cef1_08.npz_51282':{},
'reverie_qoiz87JEwZ2_38ad5b94f1f64668a5c2040a541e5ae9_16.npz_55544':{},
'reverie_mJXqzFtmKg4_a475d54e18664630b19f47575a874cf4_12.npz_55763':{},
'reverie_JmbYfDe2QKZ_a358b83b1c6f4ee3bbabb66ac93be11b_02.npz_35635':{},
'reverie_i5noydFURQK_e6ac6b13b27c41b28df5d51a7e26549d_02.npz_42367':{},
'reverie_D7N2EKCX4Sj_635e26ebc0594a23b1159bb0c6d15cc8_30.npz_56375':{},
'reverie_5LpN3gDmAk7_98ada9cf49e441539d91b3f45839539b_01.npz_18696':{},
'reverie_p5wJjkQkbXX_d99a0223c600419aa300a253164774ba_06.npz_56435':{},
'reverie_B6ByNegPMKs_6ef1007d4f8641cfa4102b27787f0210_00.npz_56557':{},
'reverie_gTV8FGcVJC9_85599a90376043339b55e424eeb9aeab_12.npz_56957':{},
'reverie_EDJbREhghzL_675491e4cc5a459db65fba26b03e6369_06.npz_57118':{},
'reverie_ur6pFq6Qu1A_2ab7cc6ff29e4f8298446de2fb2a3f01_04.npz_58053':{},
'reverie_s8pcmisQ38h_d75fa5875ff1464497e5c2191ee803b9_00.npz_58176':{},
'reverie_r1Q1Z4BcV1o_7f71bf54a3884deabc7cb90728c7e77d_04.npz_58839':{},
'reverie_jh4fc5c5qoQ_3182b6eba9ce405f9e1c0af86cc6b862_00.npz_41828':{},
'reverie_ac26ZMwG7aT_7fb0cc6c0754461ca29084c1d793b600_00.npz_58981':{},
'reverie_B6ByNegPMKs_61fa2813e92c46b8a716b697bc05756a_32.npz_59093':{},
'reverie_EDJbREhghzL_bace700b94f743d1bedc82b611604dee_01.npz_25249':{},
'reverie_D7N2EKCX4Sj_12e9fa5962de43698cf9cec57105d287_14.npz_59446':{},
'reverie_8WUmhLawc2A_b3cb0280416742fc9dcba0d2d9e5d2f7_07.npz_59864':{},
'reverie_8WUmhLawc2A_cc356636fdbf45769882f5912bc6d009_04.npz_50124':{},
'reverie_Pm6F8kyY3z2_87e7b6f2006541a9abe57fba18294a0c_02.npz_61007':{},
'reverie_D7N2EKCX4Sj_2e061585204b4ba686fd66f746243127_00.npz_61107':{},
'reverie_ac26ZMwG7aT_88fe76c2969d431caf5d60ff7aa5467a_02.npz_61326':{},
'reverie_p5wJjkQkbXX_4c82e62e485548a98cfbebff56f6c7d4_09.npz_38172':{},
'reverie_cV4RVeZvu5T_3e51eeaac8404b31ad8a950bb2bb953d_17.npz_19430':{},
'reverie_qoiz87JEwZ2_0a796daf5b734fbbb52974e2349a6c0e_14.npz_64710':{},
'reverie_759xd9YjKW5_146714339e954166a2f701202e030c29_01.npz_65867':{},
'reverie_VzqfbhrpDEA_c2ff31961cd0464d9fd12ba5208ffa52_03.npz_66420':{},
'reverie_PX4nDJXEHrG_673c93ec5cfd45efb87f064bd4723f8d_06.npz_68881':{},
'reverie_7y3sRwLe3Va_93318cd6a48a4eb59eb59d2481095044_07.npz_57442':{},
'reverie_vyrNrziPKCB_9fd63460efcf4292a96791b9531b6f94_12.npz_74922':{},
'reverie_5LpN3gDmAk7_9231543c6c484c939a501338ad8c1db4_06.npz_49458':{},
'reverie_sT4fr6TAbpF_12eaf8b2b7b64622b330bd58d275f02a_07.npz_82659':{},
'reverie_S9hNv5qa7GM_addb4d69f82243fb9a38c4d615651ba7_00.npz_34234':{},
'reverie_V2XKFyX4ASd_6ed853a792384773975309cad93b10f2_10.npz_87843':{},
'reverie_V2XKFyX4ASd_600eeada310e4c8d8e941cffd26421e8_20.npz_87888':{},
'reverie_759xd9YjKW5_0abcad1faaaa45d296aac7b2083ebe54_08.npz_65577':{},
'reverie_b8cTxDM8gDG_de07fa6aafd04c698dbc2dd9cdae9948_20.npz_88349':{},
'reverie_b8cTxDM8gDG_6268c4082bdf4f029898a13ca335a8a1_32.npz_88396':{},
'reverie_rPc6DW4iMge_d7044641458e49b0803e186bd1856994_00.npz_36069':{},
'reverie_sKLMLpTHeUy_2c0e745d95304653a5ec42b9e4abbadd_06.npz_89669':{},
'reverie_PuKPg4mmafe_4122ef1b23ca44b78c7cb26fa1c6a645_06.npz_90164':{},
'reverie_5q7pvUzZiYa_69fad7dd177847dbabf69e8fb7c00ddf_05.npz_79821':{},
'reverie_VLzqgDo317F_ba47cad0e7e748c390f8e7d1e94ccbb4_09.npz_80857':{},
'reverie_JeFG25nYj2p_91956356893c43538dfd48c4e647d531_00.npz_90771':{},
'reverie_VzqfbhrpDEA_1cb182460525496b96afe41e8bb4815a_02.npz_90922':{},
'reverie_rPc6DW4iMge_1b4824ffdfc44d08aa648e90241289e4_04.npz_59520':{},
'reverie_D7N2EKCX4Sj_767bb19c79af4193937ebe7c911a1d9e_02.npz_91991':{},
'reverie_D7N2EKCX4Sj_e3c67078918d48a8a37abdbd38c61839_04.npz_76651':{},
'reverie_D7N2EKCX4Sj_b3ea270a560d4fc784e7c7d4ca0e2248_01.npz_92078':{},
'reverie_D7N2EKCX4Sj_c533d0a4d31442048cedb9a05c762ac9_05.npz_92202':{},
'reverie_D7N2EKCX4Sj_61e6284b6ef541e59a87efa918514255_01.npz_92276':{},
'reverie_D7N2EKCX4Sj_1d7b7a08654f46df87604e7ae30f06b5_04.npz_92326':{},
'reverie_D7N2EKCX4Sj_48398dee77ab44adbe2486df3d18bfe6_03.npz_92375':{},
'reverie_cV4RVeZvu5T_c2dcba0e507748e0b4eb18690058666e_02.npz_92435':{},
'reverie_cV4RVeZvu5T_ffb7f7310ae84246a7184818cbdbba38_02.npz_92454':{},
'reverie_cV4RVeZvu5T_3ed3115dbc9d46fdbd694e3e636d50d3_04.npz_92459':{},
'reverie_cV4RVeZvu5T_1767bad8f63a4a8497c8122da77ee7d9_03.npz_92484':{},
'reverie_2n8kARJN3HM_2d21aec2e48b4797ad0367e96fd316f4_19.npz_92599':{},
'reverie_D7N2EKCX4Sj_1e013670f0ba4c9494734cedf464e11a_00.npz_75956':{},
'reverie_D7N2EKCX4Sj_aff4da2529fc495d888d0a87236e84b3_07.npz_96525':{},
'reverie_D7N2EKCX4Sj_d3a67198e8584782bdc96f2612d8e432_07.npz_96630':{},
'reverie_D7N2EKCX4Sj_c3f0a365dde24eb1a98807c5f02f8125_05.npz_96693':{},
'reverie_D7N2EKCX4Sj_94c6c71a12b243d1b011d30dd2858fbb_04.npz_97061':{},
'reverie_D7N2EKCX4Sj_027c36f1fb444f0c87bfa768d6d94930_05.npz_97172':{},
'reverie_D7N2EKCX4Sj_b07554ffc3164d45bfe9410a40632261_03.npz_97282':{},
'reverie_D7N2EKCX4Sj_4cd489ac0eb3416883212e16c15e659e_25.npz_97530':{},
'reverie_Vvot9Ly1tCj_f21573f2d2f244fd9e176c94f75b063f_15.npz_98199':{},
'reverie_rPc6DW4iMge_d51bc78ddd6b4e7ba616f70584cafd95_04.npz_98536':{},
'reverie_rPc6DW4iMge_0582a7daf9aa47f9a2d42896d70319d2_03.npz_98607':{},
'reverie_VLzqgDo317F_f0244e1e73e34c6997e25f07e9b42328_14.npz_98743':{},
'reverie_kEZ7cmS4wCh_c6957ec905d04ac8acab161452470981_04.npz_65501':{},
'reverie_D7N2EKCX4Sj_9b38aab252404f32b2820c597569cac6_05.npz_99212':{},
'reverie_D7N2EKCX4Sj_223e143862134f51b3e794c840d495c4_19.npz_99431':{},
'reverie_p5wJjkQkbXX_6d3d71725b3d40429c2e06972e5fa0e8_04.npz_91946':{},
'reverie_cV4RVeZvu5T_44762c24817a4a4fab97961ebaaee07f_07.npz_20009':{},
'reverie_D7N2EKCX4Sj_5932ea805ae44802a1d412d5d4ab61af_15.npz_95166':{},
'reverie_D7N2EKCX4Sj_95dbd6706dc047bcba49041c8368359b_13.npz_95283':{},
'reverie_D7N2EKCX4Sj_0c2d645fd3b64fee95d8f447e8efdeaf_25.npz_95511':{},
'reverie_D7N2EKCX4Sj_cfd4487d04854cbdb03680460e12d5ca_24.npz_106886':{},
'reverie_8WUmhLawc2A_71c4574785a44394bfb662c2e1c617bf_18.npz_106996':{},
'reverie_8WUmhLawc2A_44121bda3e5548c19b595e16c8e59c59_19.npz_107148':{},
'reverie_XcA2TqTSSAj_2550b846f9db46f9936e4aa6cbb7d7b3_07.npz_107386':{},
'reverie_E9uDoFAP3SH_bde29f48ab814943baf4a7193d143d6e_01.npz_107407':{},
'reverie_VzqfbhrpDEA_02fc112c72f54dad9c127dc99ad242be_00.npz_107684':{},
'reverie_VzqfbhrpDEA_bf5ac8319e8d48568b82d9f36a064969_01.npz_107801':{},
'reverie_V2XKFyX4ASd_3fd138436a1c483593a8730fddff5187_02.npz_107873':{},
'reverie_V2XKFyX4ASd_d4507886bc484283a312687dc32d3740_02.npz_107943':{},
'reverie_mJXqzFtmKg4_acc58a31d0f14934a0bfd2a518e0f298_12.npz_107997':{},
'reverie_mJXqzFtmKg4_399143f7459a4976a621eef31ac69961_12.npz_108080':{},
'reverie_sT4fr6TAbpF_c06aae190b804b759496db0b88fe4820_15.npz_77093':{},
'reverie_b8cTxDM8gDG_dcbdda428eb846a586886b87d9552d0e_05.npz_108145':{},
'reverie_VzqfbhrpDEA_497341f6deb9491d8069654da6166366_02.npz_108512':{},
'reverie_sT4fr6TAbpF_eef89e7a055443d8825cf94f092aaa35_05.npz_44370':{},
'reverie_ac26ZMwG7aT_9b4cb016c2ae494f91549fef70da01da_00.npz_98436':{},
'reverie_ac26ZMwG7aT_7eb32b1e57c644f1b5a5801dbc6401cc_00.npz_98467':{},
'reverie_cV4RVeZvu5T_ac87cf2f5d1f4c6e82be217339e7f0a6_05.npz_109670':{},
'reverie_2n8kARJN3HM_397cae1de3b74644804196b838fd3c8a_09.npz_100775':{},
'reverie_D7N2EKCX4Sj_d525db12d1924c379c74f4e198115058_06.npz_110411':{},
'reverie_D7N2EKCX4Sj_9e57fd7d4e9949f1a03f765da4a068fe_04.npz_110514':{},
'reverie_8WUmhLawc2A_550d66ef28114bef8525d3a2d6db9cd2_09.npz_111431':{},
'reverie_8WUmhLawc2A_7bb21e85f02a4d468cd1c7040e46c596_00.npz_111540':{},
'reverie_EDJbREhghzL_987ba42081144f569eac1ddea227612a_24.npz_111973':{},
'reverie_ac26ZMwG7aT_9f2ce45f125346de825f3be52f40aab0_08.npz_112400':{},
'reverie_ac26ZMwG7aT_c173f74f5a284b98b1ac17917c4c356e_00.npz_112464':{},
'reverie_ac26ZMwG7aT_94ef53dd5e024bd7a7c79c7f43731731_09.npz_112513':{},
'reverie_ac26ZMwG7aT_626cac28fb084c17a09dfb81cfe01f22_10.npz_112571':{},
'reverie_D7N2EKCX4Sj_c94a99a685884c29b5f4857fb70298e0_10.npz_113246':{},
'reverie_Uxmj2M2itWa_b11577607584497581c9433f5406c55b_08.npz_32216':{},
'reverie_S9hNv5qa7GM_6fbfaa4207454c288739caceb9512fb5_00.npz_113817':{},
'reverie_ur6pFq6Qu1A_b3495296a3cd428f9bafcca85050c313_06.npz_106686':{},
'reverie_ur6pFq6Qu1A_96b2d4806ff04e0db5a62b331f7aa96f_02.npz_41674':{},
'reverie_VFuaQ6m2Qom_52c7f299950448828ac5d63482b5430d_19.npz_105749':{},
'reverie_VFuaQ6m2Qom_d46c90414fd94b0e9b8e2d22f12eb4ac_18.npz_105851':{},
'reverie_VFuaQ6m2Qom_773f1d2ad9ed4d69933fb547167dc303_16.npz_105935':{},
'reverie_8WUmhLawc2A_cbbff2fa222241dfb2e16f3dba643e16_17.npz_116966':{},
'reverie_ur6pFq6Qu1A_a4e9fd45c60c4fda89cda4c256bb5a81_04.npz_117334':{},
'reverie_ur6pFq6Qu1A_cbd3e6decb8147e3a79d6fb9af4585a3_03.npz_117375':{},
'reverie_EDJbREhghzL_badf87714efc482f936c14de653318e5_06.npz_98349':{},
'reverie_vyrNrziPKCB_468e68b506364540972f4c718ca8c752_04.npz_118424':{},
'reverie_vyrNrziPKCB_3a0dc03705854da6abdd90b433887691_01.npz_118584':{},
'reverie_vyrNrziPKCB_d5363662598b42b5b72a401065028040_02.npz_118738':{},
'reverie_vyrNrziPKCB_74ce15df381c460a8be6fac9709c6a58_00.npz_118864':{},
'reverie_ac26ZMwG7aT_52fbea82ae7248eaaf97b1c6fbecb024_03.npz_119041':{},
'reverie_gTV8FGcVJC9_4f587b80b65145518dc6d3ee9fa3e7d6_00.npz_99574':{},
'reverie_mJXqzFtmKg4_d6bccad7baed48caa7c559073ed8b3ad_03.npz_119310':{},
'reverie_mJXqzFtmKg4_415d73ed2c93440cbc9588aef5fdecae_05.npz_119349':{},
'reverie_mJXqzFtmKg4_2cd74c32749a45f5a81ebcd9d2f4c4c7_04.npz_119421':{},
'reverie_mJXqzFtmKg4_91b191a2efb1441dafa40cabb3c91b37_05.npz_119480':{},
'reverie_mJXqzFtmKg4_39c8f7a90d6941a98bbd9e3eaafece54_02.npz_119567':{},
'reverie_D7N2EKCX4Sj_307da971edb849b99af88a99796b434d_02.npz_119613':{},
'reverie_D7N2EKCX4Sj_9f9bd1c878184026a47daf2f19e10a84_02.npz_119737':{},
'reverie_ULsKaCPVFJR_85fa57f01a904a4099aed1c7f381da81_05.npz_119828':{},
'reverie_ULsKaCPVFJR_255411360fbb4cb88190ec9bbc7f5a45_07.npz_119888':{}}
single_item_dict = {}
single_item_record = {}
sent_ids_compare = {}
import time

class ReTxtTokLmdb(TxtTokLmdb):
    def __init__(self, db_dir, max_txt_len=120):
        # load refs = [{ref_id, sent_ids, ann_id, image_id, sentences, split}]
        refs = json.load(open(f'{db_dir}/refs.json', 'r'))
        self.ref_ids = [ref['ref_id'] for ref in refs] # item_id in refs.json
        self.Refs = {ref['ref_id']: ref for ref in refs} # load all refs.json into self.Refs
        #print("^^^^^^^\n")
        #print(self.Refs)
        # load annotations = [{id, area, bbox, image_id, category_id}]
        anns = json.load(open(f'{db_dir}/annotations.json', 'r'))
        self.Anns = {ann['id']: ann for ann in anns} # load all annotations.json into self.Anns

        # load categories = [{id, name, supercategory}]
        categories = json.load(open(f'{db_dir}/categories.json', 'r'))
        self.Cats = {cat['id']: cat['name'] for cat in categories} # load all cat['name'] in categories.json into self.Cats

        # load images = [{id, file_name, ann_ids, height, width}]
        images = json.load(open(f'{db_dir}/images.json', 'r'))
        self.Images = {img['id']: img for img in images} # load all images.json into self.Images


        if max_txt_len == -1:
            self.id2len = json.load(open(f'{db_dir}/id2len.json'))
        else:
            self.id2len = {
                id_: len_
                for id_, len_ in json.load(open(f'{db_dir}/id2len.json')
                                           ).items()
                if len_ <= max_txt_len
            }
        self.max_txt_len = max_txt_len
        # self.sent_ids = self._get_sent_ids()

        self.db_dir = db_dir
        self.db = TxtLmdb(db_dir, readonly=True)
        meta = json.load(open(f'{db_dir}/meta.json', 'r'))
        self.cls_ = meta['CLS']
        self.sep = meta['SEP']
        self.mask = meta['MASK']
        self.v_range = meta['v_range']

    def _get_sent_ids(self):
        sent_ids = []
        for ref_id in self.ref_ids:
            for sent_id in self.Refs[ref_id]['sent_ids']:
                sent_len = self.id2len[str(sent_id)]
                if self.max_txt_len == -1 or sent_len < self.max_txt_len:
                    sent_ids.append(str(sent_id))

        return sent_ids

    def shuffle(self):
        # we shuffle ref_ids and make sent_ids according to ref_ids
        random.shuffle(self.ref_ids)
        self.sent_ids = self._get_sent_ids()

    def __getitem__(self, id_):
        # sent_id = self.sent_ids[i]
        txt_dump = self.db[id_]
        return txt_dump


class ReDetectFeatTxtTokDataset(DetectFeatTxtTokDataset):
    def __init__(self, txt_db, img_db):
        assert isinstance(txt_db, ReTxtTokLmdb)
        assert isinstance(img_db, DetectFeatLmdb)
        self.txt_db = txt_db
        self.img_db = img_db
        self.ids = self.txt_db._get_sent_ids()
        self.target_num = 0
        #print("@@@-----ReDetectFeatTxtTokDataset-----"+"\n")
        #print(self.ids)
        #print("@@@-----"+"\n")

    def __getitem__(self, i):
        id_ = self.ids[i]
        example = self.txt_db[id_]
        return example

    def shuffle(self):
        self.txt_db.shuffle()


class ReDataset(ReDetectFeatTxtTokDataset):
    def __getitem__(self, i):
        """
        Return:
        :input_ids     : (L, ), i.e., [cls, wd, wd, ..., sep, 0, 0]
        :position_ids  : range(L)
        :img_feat      : (num_bb, d)
        :img_pos_feat  : (num_bb, 7)
        :attn_masks    : (L+num_bb, ), i.e., [1, 1, ..., 0, 0, 1, 1]
        :obj_masks     : (num_bb, ) all 0's
        :target        : (1, )
        """
        # {sent_id, sent, ref_id, ann_id, image_id, bbox, input_ids}
        example = super().__getitem__(i)
        image_id = example['image_id']
        fname = example['npz_name']
        
        # fname = f'visual_grounding_coco_gt_{int(image_id):012}.npz'
        #img_feat, img_pos_feat, num_bb = self._get_img_feat(fname)
        img_feat, img_pos_feat, num_bb, object_id= self._get_img_feat(fname)

        # text input
        input_ids = example['input_ids']
        input_ids = self.txt_db.combine_inputs(input_ids)
        attn_masks = torch.ones(len(input_ids) + num_bb, dtype=torch.long)

        # target bbox
        img = self.txt_db.Images[image_id]
        try:
            assert len(img['ann_ids']) == num_bb, \
                'Please use visual_grounding_coco_gt'
        except:
            stop=1

        target = img['ann_ids'].index(example['ann_id'])
        target = torch.tensor([target])

        # obj_masks, to be padded with 1, for masking out non-object prob.
        obj_masks = torch.tensor([0]*len(img['ann_ids']), dtype=torch.uint8)
        self.target_num += len(target)
        frame_tag = fname + "_" + str(example['ann_id'])# + "_" + str(example['sent_id'])
        sent_ids_compare[example['sent_id']] = 1
        if len(sent_ids_compare) > 195462:
            print("example['sent_id']", example['sent_id'])
            sent_ids_file = "/src/reverie_data_generated/operations_2/compare_result/sent_ids_file.json"
            with open(sent_ids_file, "w") as f:
                json.dump(sent_ids_compare, f)
        
        #=========zcy
        sent_ids_zcy = example['sent_id']
        sent_ids_zcy = torch.tensor([sent_ids_zcy])
        ascii = np.fromstring(fname, dtype=np.uint8)
        npz_name_zcy = torch.tensor([ascii])
        #=========zcy
        
        #print("all_images", frame_tag, i/64,i)
        if frame_tag == "reverie_VzqfbhrpDEA_df42352940c8404e955eafca30659120_25.npz_30724":
            print("frame_tag" , frame_tag, i/64)
            
            if len(single_item_dict) == 0:
                single_item_dict['input_ids'] = input_ids.tolist()
                single_item_dict['img_feat'] = img_feat.tolist()
                single_item_dict['img_pos_feat'] = img_pos_feat.tolist()
                single_item_dict['attn_masks'] = attn_masks.tolist()
                single_item_dict['obj_masks'] = obj_masks.tolist()
                single_item_dict['target'] = target.tolist()
                single_item_dict['object_id'] = object_id.tolist()
            else:
                input_ids_records = single_item_dict['input_ids'] == input_ids.tolist()
                img_feat_records = single_item_dict['img_feat'] == img_feat.tolist()
                img_pos_feat_records = single_item_dict['img_pos_feat'] == img_pos_feat.tolist()
                attn_masks_records = single_item_dict['attn_masks'] == attn_masks.tolist()
                obj_masks_records = single_item_dict['obj_masks'] == obj_masks.tolist()
                target_records = single_item_dict['target'] == target.tolist()
                object_id_records = single_item_dict['object_id'] = object_id.tolist()
                single_item_record[time.time()] = [i/64, input_ids_records, img_feat_records, img_pos_feat_records, attn_masks_records, obj_masks_records, target_records, object_id_records]
            for index in single_item_record:
                print(index, single_item_record[index])

        if frame_tag in sample_dict and False:
            print(frame_tag)
            if 'frequency' not in sample_dict[frame_tag]:
                sample_dict[frame_tag]['frequency'] = []
                compare_file_data = {}
                compare_file_data['input_ids'] = input_ids.tolist()
                compare_file_data['img_feat'] = img_feat.tolist()
                compare_file_data['img_pos_feat'] = img_pos_feat.tolist()
                compare_file_data['attn_masks'] = attn_masks.tolist()
                compare_file_data['obj_masks'] = obj_masks.tolist()
                compare_file_data['target'] = target.tolist()
                sample_dict[frame_tag]['data'] = compare_file_data
                sample_dict[frame_tag]['frequency'].append(i/64)
            else:
                sample_dict[frame_tag]['frequency'].append(i/64)
                print(sample_dict[frame_tag]['data']['input_ids'] == input_ids.tolist())
                print(sample_dict[frame_tag]['data']['img_feat'] == img_feat.tolist())
                print(sample_dict[frame_tag]['data']['img_pos_feat'] == img_pos_feat.tolist())
                print(sample_dict[frame_tag]['data']['attn_masks'] == attn_masks.tolist())
                print(sample_dict[frame_tag]['data']['obj_masks'] == obj_masks.tolist())
                print(sample_dict[frame_tag]['data']['target'] == target.tolist())
                if sample_dict[frame_tag]['data']['input_ids'] != input_ids.tolist():
                    print("not_equal")
                    print(frame_tag)
                    print(print(sample_dict[frame_tag]['data']['input_ids']))
                    print(input_ids.tolist())


            for item in sample_dict:
                if 'frequency' in sample_dict[item]:
                    print(item, sample_dict[item]['frequency'])
                else:
                    print(item, [])

        #if i % 10000 == 0:
        #    print("frame_tag ", frame_tag)
        """
        if frame_tag not in fname_dict:
            compare_file_data = {}
            compare_file_data['input_ids'] = input_ids.tolist()
            compare_file_data['img_feat'] = img_feat.tolist()
            compare_file_data['img_pos_feat'] = img_pos_feat.tolist()
            compare_file_data['attn_masks'] = attn_masks.tolist()
            compare_file_data['obj_masks'] = obj_masks.tolist()
            compare_file_data['target'] = target.tolist()
            #fname_dict[frame_tag] = compare_file_data
        else:
            print("=====",i)
            print(fname_dict[fname])
            print("----")
            print(input_ids.tolist())
            print(img_feat.tolist())
            print(img_pos_feat.tolist())
            print(attn_masks.tolist())
            print(obj_masks.tolist())
            print(target.tolist())
        """

        """
        if fname == 'reverie_s8pcmisQ38h_720ca9c0b604445bab3bd731000bc5ca_01.npz':
            compare_file_name = "/src/reverie_data_generated/operations_2/compare_result/1v4/720ca9c0b604445bab3bd731000bc5ca/repeat_4.json"
            compare_file_data = json.load(open(compare_file_name, "r"))
            compare_file_data[i] = {}
            compare_file_data[i]['input_ids'] = input_ids.tolist()
            compare_file_data[i]['img_feat'] = img_feat.tolist()
            compare_file_data[i]['img_pos_feat'] = img_pos_feat.tolist()
            compare_file_data[i]['attn_masks'] = attn_masks.tolist()
            compare_file_data[i]['obj_masks'] = obj_masks.tolist()
            compare_file_data[i]['target'] = target.tolist()
            
            with open(compare_file_name, "w") as f:
                json.dump(compare_file_data, f)
        """ 

        """
        print("input_ids",input_ids)
        print("img_feat",img_feat)
        print("img_pos_feat",img_pos_feat)
        print("num_bb",num_bb)
        print("example['ann_id']",example['ann_id'])
        print("target",target)
        # print(self.target_num)
        """
        return input_ids, img_feat, img_pos_feat, attn_masks, obj_masks, target, sent_ids_zcy, npz_name_zcy, object_id


def re_collate(inputs):
    """
    Return:
    :input_ids     : (n, max_L) padded with 0
    :position_ids  : (n, max_L) padded with 0
    :txt_lens      : list of [txt_len]
    :img_feat      : (n, max_num_bb, feat_dim)
    :img_pos_feat  : (n, max_num_bb, 7)
    :num_bbs       : list of [num_bb]
    :attn_masks    : (n, max_{L+num_bb}) padded with 0
    :obj_masks     : (n, max_num_bb) padded with 1
    :targets       : (n, )
    """
    (input_ids, img_feats, img_pos_feats, attn_masks, obj_masks, targets, sent_ids_zcy, npz_name_zcy

     ) = map(list, unzip(inputs))

    txt_lens = [i.size(0) for i in input_ids]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                ).unsqueeze(0)

    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)
    targets = torch.stack(targets, dim=0)

    num_bbs = [f.size(0) for f in img_feats]
    img_feat = pad_tensors(img_feats, num_bbs)
    img_pos_feat = pad_tensors(img_pos_feats, num_bbs)
    obj_masks = pad_sequence(
        obj_masks, batch_first=True, padding_value=1)

    bs, max_tl = input_ids.size()
    out_size = attn_masks.size(1)
    gather_index = get_gather_index(txt_lens, num_bbs, bs, max_tl, out_size)

    sent_ids_zcy = torch.stack(sent_ids_zcy)

    return {'input_ids': input_ids,
            'position_ids': position_ids,
            'img_feat': img_feat,
            'img_pos_feat': img_pos_feat,
            'obj_masks': obj_masks,
            'attn_masks': attn_masks,
            'gather_index': gather_index,
            'targets': targets,
            'txt_lens': txt_lens,
            'num_bbs': num_bbs,
            'sent_ids_zcy':sent_ids_zcy,
            'npz_name_zcy':npz_name_zcy}


class ReEvalDataset(ReDetectFeatTxtTokDataset):
    def __init__(self, txt_db, img_db, use_gt_feat=True):
        super().__init__(txt_db, img_db)
        self.use_gt_feat = use_gt_feat

    def __getitem__(self, i):
        """
        Return:
        :input_ids     : (L, ), i.e., [cls, wd, wd, ..., sep, 0, 0]
        :position_ids  : range(L)
        :img_feat      : (num_bb, d)
        :img_pos_feat  : (num_bb, 7)
        :attn_masks    : (L+num_bb, ), i.e., [1, 1, ..., 0, 0, 1, 1]
        :obj_masks     : (num_bb, ) all 0's
        :tgt_box       : ndarray (4, ) xywh
        :obj_boxes     : ndarray (num_bb, 4) xywh
        :sent_id
        """
        # {sent_id, sent, ref_id, ann_id, image_id, bbox, input_ids}
        sent_id = self.ids[i]
        example = super().__getitem__(i)
        image_id = example['image_id']
        if self.use_gt_feat:
            fname = f'visual_grounding_coco_gt_{int(image_id):012}.npz'
        else:
            fname = f'visual_grounding_det_coco_{int(image_id):012}.npz'
        fname = example['h5_name']
        img_feat, img_pos_feat, num_bb, object_id = self._get_img_feat(fname)

        # image info
        img = self.txt_db.Images[image_id]
        im_width, im_height = img['width'], img['height']

        # object boxes, img_pos_feat (xyxywha) -> xywh
        obj_boxes = np.stack([img_pos_feat[:, 0]*im_width,
                              img_pos_feat[:, 1]*im_height,
                              img_pos_feat[:, 4]*im_width,
                              img_pos_feat[:, 5]*im_height], axis=1)
        obj_masks = torch.tensor([0]*num_bb, dtype=torch.uint8)

        # target box
        #tgt_box = np.array(example['bbox'])  # xywh

        #zcy
        #target = img['ann_ids'].index(example['ann_id'])
        #target = np.array(target)
        #zcy

        # text input
        input_ids = example['input_ids']
        input_ids = self.txt_db.combine_inputs(input_ids)
        attn_masks = torch.ones(len(input_ids) + num_bb, dtype=torch.long)


        #return (input_ids, img_feat, img_pos_feat, attn_masks, obj_masks,
        #        tgt_box, obj_boxes, sent_id)
        return (input_ids, img_feat, img_pos_feat, attn_masks, obj_masks,
                obj_boxes, sent_id, object_id)

    # IoU function
    def computeIoU(self, box1, box2):
        # each box is of [x1, y1, w, h]
        inter_x1 = max(box1[0], box2[0])
        inter_y1 = max(box1[1], box2[1])
        inter_x2 = min(box1[0]+box1[2]-1, box2[0]+box2[2]-1)
        inter_y2 = min(box1[1]+box1[3]-1, box2[1]+box2[3]-1)

        if inter_x1 < inter_x2 and inter_y1 < inter_y2:
            inter = (inter_x2-inter_x1+1)*(inter_y2-inter_y1+1)
        else:
            inter = 0
        union = box1[2]*box1[3] + box2[2]*box2[3] - inter
        return float(inter)/union


def re_eval_collate(inputs):
    """
    Return:
    :input_ids     : (n, max_L)
    :position_ids  : (n, max_L)
    :txt_lens      : list of [txt_len]
    :img_feat      : (n, max_num_bb, d)
    :img_pos_feat  : (n, max_num_bb, 7)
    :num_bbs       : list of [num_bb]
    :attn_masks    : (n, max{L+num_bb})
    :obj_masks     : (n, max_num_bb)
    :tgt_box       : list of n [xywh]
    :obj_boxes     : list of n [[xywh, xywh, ...]]
    :sent_ids      : list of n [sent_id]
    """
    (input_ids, img_feats, img_pos_feats, attn_masks, obj_masks,
    obj_boxes, sent_ids, object_id) = map(list, unzip(inputs))

    txt_lens = [i.size(0) for i in input_ids]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                ).unsqueeze(0)

    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)

    num_bbs = [f.size(0) for f in img_feats]
    img_feat = pad_tensors(img_feats, num_bbs)
    img_pos_feat = pad_tensors(img_pos_feats, num_bbs)
    obj_masks = pad_sequence(
        obj_masks, batch_first=True, padding_value=1)

    bs, max_tl = input_ids.size()
    out_size = attn_masks.size(1)
    gather_index = get_gather_index(txt_lens, num_bbs, bs, max_tl, out_size)


    return {'input_ids': input_ids,
            'position_ids': position_ids,
            'img_feat': img_feat,
            'img_pos_feat': img_pos_feat,
            'obj_masks': obj_masks,
            'attn_masks': attn_masks,
            'gather_index': gather_index,
            
            
            'obj_boxes': obj_boxes,
            'sent_ids': sent_ids,
            'txt_lens': txt_lens,
            'num_bbs': num_bbs,
            
            'object_id': object_id}
