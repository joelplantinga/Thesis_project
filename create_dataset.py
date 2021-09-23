import random
import pandas as pd
import numpy as np
from datetime import datetime
from collections import namedtuple

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

        # Tuple objects that are used throughout the class.
        self.Prompt = namedtuple("Prompt", 'prompt_type prompt_description prompt_weight')
        self.Room = namedtuple('Room', 'room_x room_y room_floor')
        self.User = namedtuple("User", 'user_id user_x user_y user_floor user_weight team feedback')
        self.Row = namedtuple('Row', self.User._fields + self.Room._fields + self.Prompt._fields + 
                              ('tasks_performed_day', 'team_prompts',  'goalsetting', 'date_time', 'total_weight', ))

        # User information
        self.users = self.__create_users(x_users)

        # Task information
        self.prompt_description_task = [
            ("close_window", 0.20), ("open_window", 0.20),
            ("enable_airco", 0.25), ("disable_airco", 0.25),
            ("turn_off_device", 0.15), ("enable_heating", 0.20),
            ("disable_heating", 0.20), ("enable_lightning", 0.15),
            ("disable_lightning", 0.15),
            ]
        self.prompt_description_question = [
            ("check_window", 0.10), ("check_weather", 0.05), ("check_lightning", 0.05),
            ("check_airco", 0.10), ("check_heating", 0.10),
            ]

        self.prompts = self.__create_prompts()

        # General information
        self.x_users = x_users
        self.rooms = self.__create_rooms()
        self.tasks_performed_day = 100  * [0]

        # Intervention information
        self.team_prompts = 8 * [0, -1]
        self.team_goals = 16 * [0]
    
    def __create_prompts(self):

        """
        Function that creates all prompt objects.
        
        Returns:
        ------------

        tasks : list[Prompt]
            List containing all different prompts
        """

        tasks = []

        for task in self.prompt_description_task:
            description, weight = task
            tasks.append(self.Prompt("task", description, weight))
        
        for question in self.prompt_description_question:
            description, weight = question
            tasks.append(self.Prompt("question", description, weight))

        return tasks

    def __create_rooms(self):

        """
        Creates the rooms of the building. All coordinates are midpoints.
        
        Returns:
        ------------

        rooms : list[Room]
            List containing all rooms.
        """

        rooms = []
        # Room = namedtuple('Room', ['room_x room_y room_floor'])
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

        return weight

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

            random_order = random.sample(range(0, self.x_users), self.x_users)
            room = random.choice(self.rooms)
            prompt = random.choice(self.prompts)

            user_time = time

            for x in random_order:
                
                user = self.users[x]
                row = self.Row(*user, *room, *prompt, self.tasks_performed_day[user.user_id],
                               self.team_prompts[user.team], self.team_goals[user.team], 
                               user_time, 0)

                weight = self.__calculate_weight(row, date_dep)
                row = row._replace(total_weight=weight)
                self.tasks_performed_day[row.user_id] += 1

                prompts.append(row)

                if (weight > 0.5):
                    if (row.team_prompts != -1): 
                        self.team_prompts[user.team] += 1
                        self.team_goals[user.team] += 1 / prompt_goal
                        self.team_goals[user.team] = min(self.team_goals[user.team], 1)
                    break
                else:
                    user_time += pd.Timedelta(minutes=3)
            
            time += pd.Timedelta(minutes=min_per_new_prompt)

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
        initially added for calculation purposes. It also includes the
        classification. The dataset is now ready for use in ML algorithms.
        
        Attributes:
        ------------

        df : pd.Dataframe
            Dataframe containing prompts including helper columns.
        
        Returns:
        ------------

        df : pd.Dataframe
            Stripped dataframe including classification.
        """

        df["classification"] = [
            1 if (weight >0.5) else 0 for weight in df["total_weight"]
        ]

        df = df.drop(['user_weight', 'prompt_weight', 'total_weight'], axis=1)

        df.to_csv("stripped.csv", index=False)

        return(df)


environment = Dataset(x_users=100)
data = environment.generate_prompts(period=2, min_per_new_prompt=10)
data = environment.finish_dataset(data)

