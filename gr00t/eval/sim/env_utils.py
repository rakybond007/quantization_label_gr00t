import os

from gr00t.data.embodiment_tags import EmbodimentTag


def is_groot_locomanip_env(env_name: str) -> bool:
    return env_name.startswith("gr00tlocomanip")


def is_behavior_env(env_name: str) -> bool:
    return env_name.startswith("sim_behavior_r1_pro")


def is_gr1_env(env_name: str) -> bool:
    """ensures gr1 and gr1_unified are the same embodiment tag"""
    return env_name.startswith("gr1") or env_name.startswith("gr1_unified")


def is_robocasa365_env(env_name: str) -> bool:
    return env_name.startswith("robocasa/")


def get_embodiment_tag_from_env_name(env_name: str) -> EmbodimentTag:
    if is_robocasa365_env(env_name):
        # BJ: ALINVLA 는 NEW_EMBODIMENT 태그를 사용하는 것이 기본값, 만약에 GR00T N1.6 를 RoboCasa365 에서 평가할 경우에는 IS_GR00T_N1_6_EVAL=True 로 설정하면
        # BJ: ROBOCASA_PANDA_OMRON 로 조정합니다. 
        is_gr00t_n1_6_eval = os.getenv("IS_GR00T_N1_6_EVAL", False)
        if is_gr00t_n1_6_eval:
            return EmbodimentTag.ROBOCASA_PANDA_OMRON
        else:
            return EmbodimentTag.NEW_EMBODIMENT

    if is_groot_locomanip_env(env_name):
        groot_locomanip_mappings = {
            "gr00tlocomanip_g1": EmbodimentTag.UNITREE_G1,
            "gr00tlocomanip_g1_sim": EmbodimentTag.UNITREE_G1,
            "gr00tlocomanip_g1_new": EmbodimentTag.UNITREE_G1,
        }
        return groot_locomanip_mappings[env_name.split("/")[0]]

    if is_behavior_env(env_name):
        return EmbodimentTag.BEHAVIOR_R1_PRO

    if is_gr1_env(env_name):
        return EmbodimentTag.GR1

    return EmbodimentTag(env_name.split("/")[0])
