import itertools
import random
import time
from enum import IntEnum

import pandas as pd
from mpi4py import MPI
from mpi_master_slave import Master, Slave, WorkQueue

import text_mining
import utils

Tasks = IntEnum('Tasks', 'CleanCorpus CalculateCosine')


class ParallelComputingApp:
    """
    This is my application that has a lot of work to do so it gives work to do
    to its slaves until all the work is done. There different type of work so
    the slaves must be able to do different tasks
    """

    def __init__(self, slaves):
        # when creating the Master we tell it what slaves it can handle
        self.master = Master(slaves)
        # WorkQueue is a convenient class that run slaves on a tasks queue
        self.work_queue = WorkQueue(self.master)

    def terminate_slaves(self):
        self.master.terminate_slaves()

    def __add_next_task(self, i, task=None):
        """
        we create random tasks 1-3 and add it to the work queue
        Every task has specific arguments
        """
        if task == 1:
            args = i
            data = (Tasks.CleanCorpus, args)
        elif task == 2:
            args = i
            data = (Tasks.CalculateCosine, args)
        self.work_queue.add_work(data)

    def check_slaves_for_task(self, task_to_watch):
        messages = []
        start_time = time.time()
        while not self.work_queue.done():
            #
            # give more work to do to each idle slave (if any)
            #
            self.work_queue.do_work()

            #
            # reclaim returned data from completed slaves
            #
            for slave_return_data in self.work_queue.get_completed_work():
                #
                # each task type has its own return type
                #
                task, data = slave_return_data
                if task == task_to_watch:
                    done, arg1 = data
                    if done:
                        messages.append(arg1)
        print(f"Task: {Tasks(task_to_watch).name} processing time: {(time.time() - start_time) * 1000} ms.")
        return messages

    def run(self):
        """
        Keep running slaves as long as there is work to do
        """
        movie_limit = 1000
        df = pd.read_csv("./data/netflix_titles.csv")  # netflix dataset
        texts = df["description"].to_numpy()[:movie_limit]  # slice for testing purposes

        for i in texts:
            self.__add_next_task(i, Tasks.CleanCorpus)

        tokenized_documents = self.check_slaves_for_task(Tasks.CleanCorpus)
        vocab = set(list(itertools.chain.from_iterable(tokenized_documents)))
        print("Vocabulary length: ", len(vocab))

        vectors = text_mining.get_tf_idf_vectors(vocab, tokenized_documents)
        index = random.randint(0, 100)
        print(f"random index for movie: {index}")
        chosen_movie = vectors.pop(index)

        for vector_index, vector in enumerate(vectors):
            self.__add_next_task((chosen_movie, vector_index, vector), Tasks.CalculateCosine)

        similarity_vectors = self.check_slaves_for_task(Tasks.CalculateCosine)

        result_similarity = text_mining.get_similarity_items(n=3, similarity=similarity_vectors)
        movie_items_index = [item.get("index") for item in result_similarity]

        utils.show_recommendation(df, index, movie_items_index)


class MySlave(Slave):
    """
    Class responsible for executing tasks in parallel.
    It is important to note that the execution is done given a type of task.
    """

    def __init__(self):
        super(MySlave, self).__init__()

    def do_work(self, args):
        task, data = args  # the data contains the task type

        rank = MPI.COMM_WORLD.Get_rank()
        name = MPI.Get_processor_name()

        # Every task type has its specific data input and return output
        ret = None
        if task == Tasks.CleanCorpus:
            arg1 = data
            print(f'Slave {name} rank {rank} executing {task} task_id')
            ret = (True, text_mining.pipeline_cleaning(arg1))

        elif task == Tasks.CalculateCosine:
            movie, vector_index, vector = data
            arg1 = {
                "index": vector_index,
                "distance": text_mining.cosine_similarity([movie], [vector])
            }
            ret = (True, arg1)

        return task, ret


def main():
    name = MPI.Get_processor_name()
    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()

    print('I am  %s rank %d (total %d)' % (name, rank, size))

    if rank == 0:  # Master

        app = ParallelComputingApp(slaves=range(1, size))
        app.run()
        app.terminate_slaves()

    else:  # Slaves
        MySlave().run()

    print(f'Task completed (rank {rank})')


if __name__ == "__main__":
    main()
