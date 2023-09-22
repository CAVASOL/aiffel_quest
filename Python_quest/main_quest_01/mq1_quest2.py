import random as r

class Aiffel:
    ids = 0
    core = {"강임구", "김대선", "김연", "김영진", "박해극", "서승호", "송민찬", "오선우", "이혁희",
            "전다빈", "정인호", "조세창", "조필선", "지동현", "최재혁", "최현우", "강동원"}

    def __init__(self, name, year, generation, course):
        self.name = name
        self.year = year
        self.generation = generation
        self.course = course
        self.id = self.gen_id()
        self.quest_score = 0
        self.total_score = 0
        self.total_penalty = 0
        self.group = self.random_group()

    @classmethod
    def get_id_num(cls):
        return cls.ids

    def score(self):
        input_score = int(input("Please enter your score. ** Scores range between 1 and 3 ** : "))
        if 1 <= input_score <= 3:
            self.total_score += input_score
        else:
            print("\nInvalid input. Score should be between 1 and 3.")

    def penalty(self):
        input_penalty = int(input("\nPlease enter the penalty score to be deducted. ** Penalty scores range between 1 and 3 ** : "))
        if 1 <= input_penalty <= 3:
            self.total_penalty += input_penalty
        else:
            print("\nInvalid input. Penalty score should be between 1 and 3.")

    def gen_id(self):
        year = str(self.year)[-2:]

        if self.generation == "온라인6기":
            num = "27"
        else:
            num = "25"

        if self.course != "코어":
            course = "2"
        else:
            course = "1"

        list_abc = list(sorted(Aiffel.core))
        num_abc = str(list_abc.index(self.name) + 1).zfill(2)

        aiffel_id = year + num + course + num_abc
        Aiffel.ids += 1
        return aiffel_id

    def random_group(self):
        return r.choice(Aiffel_Group.groups)

    def display_info(self):
        formatted_quest = '{:,}'.format(self.quest_score)
        print(f'\n[ 이름: {self.name}, 학번: {self.id}, 과정종류: {self.course}, '
              f'퀘스트 점수 총점: {self.total_score}, 퀘스트 입력 총점: {formatted_quest}, '
              f'퀘스트 penalty 총합: {self.total_penalty}, 그룹: {self.group} ]')

    @classmethod
    def remove_from_core(cls, name):
        name = str(input("\nHave anyone given up? Please enter the name. : "))
        if name in cls.core:
            cls.core.remove(name)
            print(f'\n{name} removed from the list.')
        else:
            print(f'\n{name} is not in the list.')

class Aiffel_Group(Aiffel):
    groups = ["상생", "신뢰", "열정", "이끔", "성장", "가치", "공유", "확산"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.group = self.random_group()

    @classmethod
    def random_grouping(cls, core_list):
        groups = cls.groups
        group_size = [2, 3, 4]
        grouped_students = []

        for size in group_size:
            while len(core_list) >= size:
                group = r.sample(core_list, size)
                for student in group:
                    core_list.remove(student)
                group_name = r.choice(groups)
                grouped_students.append((group_name, group))

        while len(core_list) > 0:
            size = r.choice(group_size)
            group = r.sample(core_list, size)
            for student in group:
                core_list.remove(student)
            group_name = r.choice(groups)
            grouped_students.append((group_name, group))

        return grouped_students

    def display_info(self):
        super().display_info()
        print(f'\nGroup: {self.group}')

    def display_grouping(self):
        core_list = list(Aiffel.core)
        grouped_students = self.random_grouping(core_list)
        for group_name, group in grouped_students:
            print(f'\nGroup: {group_name}')
            for student in group:
                print(f'{student}')

class Aiffel_Guild(Aiffel):
    guilds = ["Hinton", "Altman", "Ng", "Hassabis"]
    guild_scores = {"Hinton": 0, "Altman": 0, "Ng": 0, "Hassabis": 0}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.guild = self.random_guild()

    def random_guild(self):
        return r.choice(Aiffel_Guild.guilds)

    def guild_score(self):
        print(f"\nAiffel Core Course consists of the following guilds - Hinton | Altman | Ng | Hassabis")
        guild_name = input("\nEnter the guild name. : ")
        if guild_name in Aiffel_Guild.guilds:
            score = int(input(f"\nEnter the score for {guild_name}. ** Scores range between 1 and 3 ** : "))
            Aiffel_Guild.guild_scores[guild_name] += score
        else:
            print("\nInvalid guild name.")

    def guild2guild_score(self, guild_name, score1):
        print(f"\nIs there a guild you would like to give special score to? - Hinton | Altman | Ng | Hassabis")
        guild_name = input("\nEnter the guild name for special score! : ")
        if guild_name in Aiffel_Guild.guilds:
            score = int(input(f"\nEnter the score for {guild_name}. ** Scores range between 1 and 3 ** : "))
            Aiffel_Guild.guild_scores[guild_name] += score1
        else:
            print("\nInvalid guild name.")

    def display_info(self):
        super().display_info()
        for guild_name, score in Aiffel_Guild.guild_scores.items():
          print(f'\n ** Guild {guild_name}: {score}')


grew = Aiffel("김연", 2023, "온라인6기", "코어")
grew.score()
grew.penalty()
grew.display_info()
grew.remove_from_core("강동원")

group_grew = Aiffel_Group("김연", 2023, "온라인6기", "코어")
group_grew.display_info()

guild_grew = Aiffel_Guild("김연", 2023, "온라인6기", "코어")
guild_grew.guild_score()
guild_grew.guild2guild_score("Hinton", 1)
guild_grew.display_info()