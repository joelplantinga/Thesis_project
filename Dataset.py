import random
from collections import namedtuple
from datetime import datetime
from namedlist import namedlist #needs install
import numpy as np
import pandas as pd
import math


class Dataset:
    
    """
    Class that makes the dataset.

    Attributes:
    ------------

    x_users : int
        The number of users that are generated in the environment
        (default is 4)

    Methods:
    ------------

    generate_prompts(period=365, min_per_new_prompt=10, date_dep=True)
        generates a sample dataset of the system ran over a given period.

    """

    def __init__(self, x_users=100):
        
        """
        Parameters:
        ------------

        x_users : int
            The number of users that are generated in the environment
            (default is 100)
        """

        # Objects that are used throughout the class.
        self.Prompt = namedtuple("Prompt", 'prompt_type prompt_description prompt_weight device has_location')
        self.Room = namedtuple('Room', 'room_x room_y room_floor')
        self.User = namedtuple("User", 'user_id user_x user_y user_floor user_weight team feedback')
        self.Device = namedtuple("Device", 'device weight type')
        self.State = namedlist("State", "window airco heating lightning device") # must be mutable

        self.Row = namedtuple('Row', self.User._fields + self.Room._fields + self.Prompt._fields + 
                              ('tasks_performed_day', 'team_prompts',  'goalsetting', 'date_time', 'total_weight', 'classification',))

        # User information
        self.users = self.__create_users(x_users)

        self.devices = [
            self.Device("window", 0.20, 1), self.Device("airco", 0.25, 1),
            self.Device("heating", 0.20, 1), self.Device("lightning", 0.15, 1),
            self.Device("device", 0.15, 2),
            ]
        

        # General information
        self.x_users = x_users
        self.rooms = self.__create_rooms()
        self.tasks_performed_day = 100  * [0]

        self.states = self.__create_states(len(self.rooms))
        # self.prompts = self.__create_prompts()

        # Intervention information
        self.team_prompts = 8 * [0, -1]
        self.team_goals = 16 * [0]

    def __create_states(self, x_rooms):

        states = []
        for room in range(x_rooms):
            states.append(self.State(
                random.randint(0, 1), random.randint(0, 1),
                random.randint(0, 1), random.randint(0, 1),
                random.randint(0, 1)
            ))
        
        # print(*states, sep="\n")
        return states

    def __select_prompt(self, room_nr):
        
        # 10% change of getting a weather question.
        if(random.randint(1,10) == 1):
            return self.Prompt("question", "check weather" , 0.05, "no_device", 0)

        dev = random.choice(self.devices)
        state = getattr(self.states[room_nr], dev.device)
        type = dev.type

        # 1 out 2 change of getting a question vs a task.
        if(random.randint(0, 1)):
            if(state):
                return self.Prompt("task", "turn off " + dev.device, dev.weight, dev.device, 1)
            else: 
                if(type == 1):
                    return self.Prompt("task", "turn on " + dev.device, dev.weight, dev.device, 1)
                else:
                    return self.Prompt("question", "check " + dev.device, dev.weight-0.1, dev.device, 1)
        else:
            return self.Prompt("question", "check " + dev.device, dev.weight-0.1, dev.device, 1)

    def __create_rooms(self):

        """
        Creates the rooms of the building. All coordinates are midpoints.
        
        Returns:
        ------------

        rooms : list[Room]
            List containing all rooms.
        """

        rooms = []
        for floor in range(8):
            for x in range(10, 91, 80):
                rooms.append(self.Room(x, 20, floor))
            
            for x in range(30, 71, 20):
                for y in range(10, 31, 20):
                    rooms.append(self.Room(x, y, floor))

        return rooms

    def __create_users(self, x_users):
        
        """
        Creates the users of the organization.

        Attributes:
        ------------

        x_users : int
            The number of users that are generated in the environment
            (default is 4)

        Returns:
        ------------

        users : list[User]
            List containing all rooms.
        """

        users = []

        for id, user in enumerate(range(x_users)):

            floor = random.randint(0, 7)
            x = round(random.uniform(0, 100), 2)
            team = floor * 2 + (x > 50)

            users.append(self.User(id, x, round(random.uniform(0, 40), 2),
                                   floor, round(random.uniform(0, 1), 3),
                                   team, random.randint(0, 1)))

        # print(*users, sep="\n")
        return users               

    def __calculate_weight(self, row, date_dep):

        """Calculating the changes that a task will be performed using a static
        formula. 

        Attributes:
        ------------

        row : Row object
            A row from the dataset containing all information about the prompt.
        
        min_per_new_prompt : int
            Parameter that indicates the time between the newly generated prompts. 
        
        period : int
            the timespan (in days) over the dataset will be created.
        
        date_dep : bool
            Indicates whether performance indication will be dependent on the date
            or not.

        Returns:
        ------------

        weight : float
            Likelihood that the prompt will be performed by the user.

        """

        weight = 1  # initial weight set to 1

        weight *=  ((row.user_weight + 1) / 2)

        dist = pow(pow(row.room_x - row.user_x, 2) +
                   pow(row.room_y - row.user_y, 2), 0.5)
        weight *=  (1 - (dist / 108 * 0.25))  # 108 =~ max diff in dist

        weight *=  (1 - (abs(row.user_floor - row.room_floor) * (1/7) * 0.6))

        weight *=  (1 - (row.tasks_performed_day * 0.02))

        weight *=  (1 - row.prompt_weight)

        weight *=  1  # team does not influence the weight

        
        # Dependency on the dates
        if(date_dep):
            start_date = datetime.strptime("2021-01-01", "%Y-%m-%d")
            date_diff = (row.date_time - start_date).days

            weight = weight * (1 - (date_diff / 10000))
            
            season = abs(6 - min(row.date_time.month, 12-row.date_time.month))
            weight = weight * (1.09 - (season * 0.03))

        if(row.feedback):
            weight *= 1.10
        
        if(row.team_prompts != -1):
            # People tend to lose motivation nearing the goal
            weight *= 1 + (0.1 * (1 - row.goalsetting))

        

        return min(0.99, weight)

    def __update_state(self, room_nr, dev):
        
        state = getattr(self.states[room_nr], dev)
        setattr(self.states[room_nr], dev, not(state))

    def __get_user_room(self, user):
        
        if (user.user_x > 80):
            x = 90
            y = 20
        elif (user.user_x <= 20):
            x = 10
            y = 20
        else:
            x = int(20 * math.ceil(float(user.user_x)/20)) - 10
            y = int(20 * math.ceil(float(user.user_y)/20)) - 10

        return self.Room(x, y , user.user_floor)

    def __generate_prompts_day(self, time, min_per_new_prompt, date_dep, prompt_goal):
        
        """Function that generates prompt for 1 day. Prompts are generated 
        from 8.00 and stop generating no later than 19.00. The time between
        newly generated prompts can be modified using the parameter. 
        
        Attributes:
        ------------

        time : Timestamp object
            Indicates the date and time of the day. Prompts will be generated 11 
            hours from then.
        
        min_per_new_prompt : int
            Parameter that indicates the time between the newly generated prompts. 
        
        period : int
            The timespan (in days) over the dataset will be created.
        
        date_dep : bool
            Indicates whether performance indication will be dependent on the date
            or not.
        
        prompt_goal : int
            Indicates the amount of tasks that should be performed in 1 year to pass
            the goal.

        Returns:
        ------------

        prompts : pd.Dataframe
            Dataframe containing prompts of one day. 
        """
        prompts = []
        
        while (time.hour < 19):
            
            # TODO: make sure that roomNR is available and that states get updates after the task is performed.
            # TODO: make sure that weather questions have no room nr and no XY coords.
            random_order = random.sample(range(0, self.x_users), self.x_users)
            
            room_nr = random.randrange(0, len(self.rooms))
            room = self.rooms[room_nr]
            
            # room = random.choice(self.rooms)
            prompt = self.__select_prompt(room_nr)

            user_time = time

            for x in random_order:
                
                user = self.users[x]

                # tasks without room can be performed in the room of the user
                if(not(prompt.has_location)): room = self.__get_user_room(user)

                row = self.Row(*user, *room, *prompt, self.tasks_performed_day[user.user_id],
                               self.team_prompts[user.team], self.team_goals[user.team], 
                               user_time, 0, 0)

                weight = self.__calculate_weight(row, date_dep)
                row = row._replace(total_weight=weight)
                
                classification = random.uniform(0, 1) <= weight
                row = row._replace(classification=classification)


                prompts.append(row)

                if (classification == True):

                    self.tasks_performed_day[row.user_id] += 1

                    if(prompt.prompt_type == "task"):
                        self.__update_state(room_nr, prompt.device)
                        # TODO: update statefunction, get roomnumber, get device
                    if (row.team_prompts != -1): 
                        self.team_prompts[user.team] += 1
                        self.team_goals[user.team] += 1 / prompt_goal
                        self.team_goals[user.team] = min(self.team_goals[user.team], 1)
                    break
                else:
                    user_time += pd.Timedelta(minutes=3)
            
            time += pd.Timedelta(minutes=min_per_new_prompt)

        for state in self.states:
            state.device = 1

        return pd.DataFrame(prompts)

    def generate_prompts(self, period=365, min_per_new_prompt=10, date_dep=True):
        
        """Function that generates a dataset consisting of prompts that are
        pushed to users of the system. 

        NOTE: The goalsetting effect is calculated in a such a way that every team could
        meet the goal if the tasks where equally performed over the teams. 


        Attributes:
        ------------
        
        period : int
            The timespan (in days) over the dataset will be created.

        min_per_new_prompt : int
            Parameter that indicates the time between the newly generated prompts. 
                
        date_dep : bool
            Indicates whether performance indication will be dependent on the date
            or not.
        
        Returns:
        ------------

        prompts : pd.Dataframe
            Dataframe containing prompts over the given period. 
        """

        prompts = pd.DataFrame()
        dates = pd.date_range("2021-01-01 08:00:00.000000", periods=period).tolist()

        # Amount of tasks that should be performed to pass the goal.
        # ((Minutes per day / time between newly generated tasks) * year) / teams
        prompt_goal = (((11 * 60) / min_per_new_prompt) * 365) / 16


        for date in dates:
                        
            self.tasks_performed_day = 100  * [0] # Reset task count

            if (date.day == 1 and date.month == 1):
                self.team_prompts = 8 * [0, -1]

            prompts = prompts.append(self.__generate_prompts_day(date, min_per_new_prompt, date_dep, prompt_goal))
        
        prompts.to_csv("full.csv", index=False)
        
        return(prompts)

    def finish_dataset(self, df):

        """
        Function that strips the dataset with the variables that were
        initially added for calculation purposes. The dataset is now ready 
        for use in ML algorithms.
        
        Attributes:
        ------------

        df : pd.Dataframe
            Dataframe containing prompts including helper columns.
        
        Returns:
        ------------

        df : pd.Dataframe
            Stripped dataframe including classification.
        """

        df = df.drop(['user_weight', 'prompt_weight', 'total_weight'], axis=1)

        df.to_csv("stripped.csv", index=False)

        return(df)


