"""robocasa v8 계산 사실 = v7 과 같고 **기간 주장만 뺐다**.

v7 은 "over the next ~1 second" 라고 적는데 창 16스텝은 20fps 에서 0.80초다.
나머지는 robocasa_descriptors 와 글자까지 같다 -- 여기서 다시 적지 않는다.
"""
from robocasa_descriptors import *  # noqa: F401,F403
from robocasa_descriptors import facts_text as _v7_facts


def facts_text(x):
    return _v7_facts(x).replace("over the next ~1 second", "over the chunk ahead")
