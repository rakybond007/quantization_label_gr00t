"""DexJoCo format v3; upstream 2d5b579, RC v22 / Libero v3d / HumanData v3.
User-selected no-candidate-speed facts. Four existing DexJoCo axes preserved.
"""
REVISION = '2d5b5790f85aaf12ccc8f892425bfad7a316f7a7'
GUIDANCE = "You are judging one moment of a robot's motion with a dexterous hand, to decide whether the arm could get through the stretch of motion ahead of it faster -- reaching the same places in fewer and longer steps -- without changing the outcome.\n\nGoing faster leaves the arm fewer chances to correct itself along the way. Changing the working configuration of a held object and directing a held tool onto a localized target depend on the relation between the fingers, object and target. Maintaining a supported press and carrying an already grasped object through clear space impose different constraints. Use the visible interaction to judge these checks; do not infer contact or a secure grasp from commanded motion alone.\n\nJudge the moment in front of you, not the task as a whole. One task passes through different interactions from one moment to the next."
ASK_PREFIX = 'The measurements above are stated as fact -- do not re-estimate or repeat them. Answer each check from what the cameras show about the MOMENT in front of you, read together with those measurements.\nAnswer each check on its own line as "A) 3", in order, nothing else -- one digit from 1 to 5 per check, rating how far that check describes this moment:\n  5 = clearly true of this moment\n  4 = mostly true\n  3 = partly true\n  2 = barely true\n  1 = not true at all\nA grade refers only to the check on that line.\n'
AXES = 'A) Are the fingers operating movable parts of the object being held—closing\n   jaws or folding an articulated part—so that changing the relation between\n   those parts is the operation, rather than merely holding the object?\nB) Is a held tool being directed into an impact on a localized target, rather\n   than making a sustained supported press? Use visible held-tool/target geometry together with the command motion to judge\n   preparation for or execution of a targeted strike. Command targets do not establish\n   that an impact actually occurred; the presence of a hammer alone is not enough.\nC) Is a finger or hand pressing a target supported by the surrounding surface,\n   without lifting it, striking it with a tool, or manipulating movable parts\n   within a grasp? A surface supports the target; do not assume it is bolted down.\nD) Is an already grasped object being moved or tilted as a whole through clear\n   space, without changing the working configuration of its movable parts,\n   acquiring/releasing the grasp, or entering a visibly tight target region?\n'
ASK = ASK_PREFIX + AXES + "\nAnswer:"


def normalize_facts(facts):
    lines=facts.splitlines()
    assert lines[0].startswith('COMPUTED FROM THE RECORDED ABSOLUTE ACTION TARGETS')
    assert len(lines)==4, 'Use the selected no-demand facts, not historical long facts'
    return ('MEASURED FROM THE PLANNED MOTION over the chunk ahead '
            '(these are computed facts, not estimates):\n'+'\n'.join(lines[1:]))


def build_messages(images,instruction,facts,views):
    assert len(images)==len(views)==2
    assert '\n' not in instruction
    view_note=('You are shown 2 camera views of this one moment: image 1 is the scene view ('
               +views[0]+'), and image 2 is a wrist (eye-in-hand) close-up ('+views[1]+'). '
               'The wrist camera is mounted on the hand, so objects normally look close in it '
               '-- general closeness is normal. Use the wrist view to judge the close moments: '
               'the fingers operating the held object, a tool approaching its target, and '
               'a finger pressing a supported target.')
    text='\n\n'.join([GUIDANCE,view_note,'The robot was told: '+instruction,normalize_facts(facts),ASK])
    m=[{'role':'user','content':[{'type':'image','image':im} for im in images]+[{'type':'text','text':text}]}]
    validate(m)
    return m


def validate(messages):
    assert len(messages)==1 and messages[0]['role']=='user'
    c=messages[0]['content'];assert [x['type'] for x in c]==['image','image','text']
    t=c[-1]['text']
    assert t.startswith(GUIDANCE+'\n\n') and t.endswith(ASK)
    assert t.count('The robot was told: ')==1
    assert t.index('The robot was told: ')<t.index('MEASURED FROM')<t.index(ASK_PREFIX)
    for bad in ('2x','3x','4x','YES or NO','Additional learned guidance','one second','block-last compression'):
        assert bad not in t,bad


def text_dump(messages):
    return '\n\n'.join('['+m['role'].upper()+']\n'+'\n'.join(
        x['text'] if x['type']=='text' else '[IMAGE]' for x in m['content']) for m in messages)+'\n'


# Postprocessing metadata; never inserted into the VLM prompt.
NGRADE = 5
SIGN = {"A": -1, "B": -1, "C": 1, "D": 1}
WEIGHT = {"A": 2/3, "B": 1/3, "C": 1/3, "D": 2/3}
