import csv

import model
import predictor
import itertools, os, sys
import matplotlib.pyplot as plt
from task import Task
import numpy as np


def write_results_to_csv(workflow_name, task, predictor_name, train_percent, wastage, number_train, number_test,
                         executions, seed, execution_time, number_segments):
    filename = "../results/results_" + workflow_name + "_" + str(number_segments) + ".csv"

    if not (os.path.exists(filename)):
        with open(filename, 'a', newline='\n') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(
                ["Task", "Predictor", "Setup", "Wastage", "NumberTrain", "NumberTest", "Executions", "Seed",
                 "ExecutionTime", "NumberSegments"])

    with open(filename, 'a', newline='\n') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(
            [task, predictor_name, train_percent, wastage, number_train, number_test, executions, seed, execution_time,
             number_segments])


def plot_prediction(predictor, taskinstance):
    predictor.get_model(taskinstance).plot(color="green")
    plt.scatter(range(len(taskinstance.mem)), taskinstance.mem)
    plt.show()


def plot_segstart_model(predictor, segidx, highlight=None, segcount=predictor.default_num_segments):
    plt.scatter([t.insize for t in predictor.instances],
                [[i for i in itertools.accumulate(starts)][segidx] for (starts, peaks, _) in predictor.segments])
    plt.plot([predictor.instances[0].insize, predictor.instances[-1].insize],
             predictor.endpred[segidx].predict([[predictor.instances[0].insize], [predictor.instances[-1].insize]]),
             color="green")
    if highlight != None:
        plt.scatter([highlight.insize],
                    [[i for i in itertools.accumulate(predictor.get_segments(highlight.mem, segcount)[0])][segidx]],
                    color="red")
    plt.show()


def plot_peak_model(predictor, segidx, highlight=None, segcount=predictor.default_num_segments):
    plt.scatter([t.insize for t in predictor.instances], [peaks[segidx] for (starts, peaks, _) in predictor.segments])
    plt.plot([predictor.instances[0].insize, predictor.instances[-1].insize],
             predictor.peakpred[segidx].predict([[predictor.instances[0].insize], [predictor.instances[-1].insize]]),
             color="green")
    if highlight != None:
        plt.scatter([highlight.insize], [predictor.get_segments(highlight.mem, segcount)[1][segidx]], color="red")
    plt.show()


def plot_segmentation(taskinstance, num_segments=predictor.default_num_segments):
    segments = predictor.get_segments(taskinstance.mem, num_segments)
    segends = [i for i in itertools.accumulate(segments[0])]
    segpeaks = segments[1]
    model.Model(segends, segpeaks).plot(color="green")
    plt.scatter(range(len(taskinstance.mem)), taskinstance.mem)
    plt.show()


def simulate_task(task, predictors, trainpercent, seed=18181, numsegments=predictor.default_num_segments):
    (train, test) = task.split(trainpercent, seed)
    res = {}
    for name, pred in [(n, p(train.instances, num_segments=numsegments)) for n, p in predictors.items()]:
        execs = 0
        waste = 0.0
        time = 0.0
        for t in test.instances:
            (e, w, t) = pred.get_model(t).simulate(t.mem)
            execs += e
            waste += w
            time += t
        # res[name] = (float(execs) / len(test.instances), float(waste) / len(test.instances), float(time) / len(test.instances))
        res[name] = (float(execs), float(waste), float(time))

    return res


def simulate_workflow(wfdir, numsegs=predictor.default_num_segments, trainprecenteges=[0.25, 0.5, 0.75],
                      seeds=range(10)):
    print('Reading trace data...', file=sys.stderr)

    tasks = dict([(taskname, Task(taskname, f'{wfdir}/{taskname}')) for taskname in os.listdir(f'{wfdir}/') if
                  os.path.isdir(f'{wfdir}/{taskname}')])
    predictors = dict([
        ("KS+", predictor.Predictor),
        ("Static Selective", predictor.UnipredictorSelective),
        ("Static Partial", predictor.Unipredictor),
        ("k-Segments Selective", predictor.Unipredictor2Selective),
        ("k-Segments Partial", predictor.Unipredictor2),
        ("Tovar", predictor.TovarPredictor),
        ("Tovar-Improved", predictor.TovarImprovedPredictor),
        ("Witt", predictor.WittPredictor),
        ("Default", predictor.DefaultPredictor)
    ])

    print('-----------------------------------', file=sys.stderr)
    print('Starting simulation:', file=sys.stderr)
    print('  Train/Test splits: ', trainprecenteges, file=sys.stderr)
    print('  Number of seeds: ', len(seeds), file=sys.stderr)
    print('  Number of methods: ', len(predictors), file=sys.stderr)
    print('-----------------------------------', file=sys.stderr)
    print(
        'task_name, train_rate, seed, method, num_train_instances, num_test_instances, task_executions, memory_wastage, execution_time')

    simulations = len(predictors) * len(trainprecenteges) * len(tasks) * len(
        seeds)  # sum(len(t.instances) for t in tasks.values())
    cursimulation = 0
    res = dict()

    for name, task in tasks.items():
        # if len(task.instances) < 16:
        #    print(f'Task \'{name}\' skipped due to too few executions. ({len(task.instances)})', file=sys.stderr)
        #    cursimulation += len(seeds) * len(trainprecenteges) * len(predictors)
        #    continue
        if len([1 for t in task.instances if len(t.mem) > 60]) < 16:
            print(f'Task \'{name}\' skipped due to too few executions. ({len(task.instances)})', file=sys.stderr)
            cursimulation += len(seeds) * len(trainprecenteges) * len(predictors)
            continue

        print(f'Task \'{name}\':', file=sys.stderr)
        print(f'{cursimulation}/{simulations}', file=sys.stderr, end='\r')

        res[name] = dict()
        for perc in trainprecenteges:
            print(f'  Training data: {perc * 100}% ({int(perc * len(task.instances))} samples)', file=sys.stderr)
            print(f'{cursimulation}/{simulations}', file=sys.stderr, end='\r')

            execs = dict([(n, 0.0) for n in predictors])
            waste = dict([(n, 0.0) for n in predictors])
            time = dict([(n, 0.0) for n in predictors])
            res[name][perc] = dict()
            for s in seeds:
                for (pname, (e, w, t)) in simulate_task(task, predictors, perc, s, numsegs).items():
                    execs[pname] += e
                    waste[pname] += w
                    time[pname] += t
                    cursimulation += 1
                    num_train = int(perc * len(task.instances))

                    print(
                        f'{name}, {perc}, {s}, {pname}, {num_train}, {len(task.instances) - num_train}, {e}, {w}, {t}')
                    write_results_to_csv(defaultpath, name, pname, perc, w, num_train, len(task.instances) - num_train,
                                         e, s, t, numsegs)
                    print(f'{cursimulation}/{simulations}', file=sys.stderr, end='\r')
            for pname in execs:
                exec = execs[pname] / len(seeds)
                wast = waste[pname] / len(seeds)
                tim = time[pname] / len(seeds)
                res[name][perc][pname] = (exec, wast, tim)

                print(f'  {pname}:  {exec}, {wast}, {tim}', file=sys.stderr)
            print(f'{cursimulation}/{simulations}', file=sys.stderr, end='\r')

    print('-----------------------------------', file=sys.stderr)
    print('              Summary              ', file=sys.stderr)
    print('-----------------------------------', file=sys.stderr)
    for perc in trainprecenteges:
        predres = dict()
        print(f'{perc * 100}% training data:', file=sys.stderr)
        for pname in predictors:
            exec = 0.0
            wast = 0.0
            time = 0.0
            for task in res:
                (e, w, t) = res[task][perc][pname]
                exec += e
                wast += w
                time += t
            print(f'  {pname}: {exec}, {wast}, {time}', file=sys.stderr)


defaultpath = "eager"  # sarek

simulate_workflow(f"../k-Segments-traces/{defaultpath if len(sys.argv) <= 1 else sys.argv[1]}", 4)
