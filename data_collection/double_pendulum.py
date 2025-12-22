import hebi
import datetime, time

import pandas as pd
import numpy as np
import time

START_DELAY = 0.5
SAMPLE_DELAY = 0.01
SAFETY_LOCK = True
TIME_MAX = 5.0



def setup():
    lookup = hebi.Lookup()
    time.sleep(START_DELAY)

    # Correct for your version:
    entry_list = lookup.entrylist

    for entry in entry_list:
        print("Family:", entry.family, " Name:", entry.name)

    group_shoulder = lookup.get_group_from_names(["X5-1"], ["Shoulder"])
    group_elbow = lookup.get_group_from_names(["X5-1"], ["Elbow"])

    if group_shoulder is None or group_elbow is None:
        raise AssertionError("Got None for one or more joints")

    feedback_shoulder = hebi.GroupFeedback(group_shoulder.size)
    feedback_elbow    = hebi.GroupFeedback(group_elbow.size)

    return group_shoulder, group_elbow, feedback_shoulder, feedback_elbow


if __name__ == '__main__':
    print("BEGIN setup")
    time.sleep(0.5)
    print("Ready?")
    time.sleep(0.5)
    print("Set.")
    time.sleep(0.5)
    print("Go!")

    group_shoulder, group_elbow, feedback_shoulder, feedback_elbow = setup()
    print("END setup")
    
    t0 = time.time()

    print("BEGIN loop")
    df = pd.DataFrame(

    )

    columns = ['time']
    for joint in ('shoulder', 'elbow'):
        for var in ('qpos', 'qvel', 'effort'):
            columns.append(f'{joint}_{var}')
    print("Columns =", columns)
    df = pd.DataFrame(columns=columns)
    print(df)

    while time.time() - t0 < TIME_MAX:
        feedback_shoulder = group_shoulder.get_next_feedback(reuse_fbk=feedback_shoulder)
        feedback_elbow = group_elbow.get_next_feedback(reuse_fbk=feedback_elbow)

        t = time.time() - t0
        print("time:", t)
        # print("Shoulder qpos:", feedback_shoulder.position)
        # print("Shoulder qvel:", feedback_shoulder.velocity)
        # print("Shoulder effort:", feedback_shoulder.effort)
        # print("Shoulder gradient vel:", np.gradient(feedback_shoulder.position, time[:,0], axis=0))
        # print("Shoulder gradient acc:", np.gradient(feedback_shoulder.velocity, time[:,0], axis=0))
        # print("ELbow qpos:", feedback_elbow.position)
        # print("Elbow qvel:", feedback_elbow.velocity)

        df.loc[len(df)] = ( # NOTE each is a len 1 list
            t,
            feedback_shoulder.position[0],
            feedback_shoulder.velocity[0],
            feedback_shoulder.effort[0],
            feedback_elbow.position[0],
            feedback_elbow.velocity[0],
            feedback_elbow.effort[0],
        )

        print(feedback_shoulder.accelerometer)

        time.sleep(SAMPLE_DELAY)

    print("END loop")
    datetime_str = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    df.to_csv(f'/accelerometer_test.csv')



