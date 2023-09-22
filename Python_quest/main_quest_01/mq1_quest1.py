import random as r

class Account:
    bank = "SC은행"
    vip = []
    accounts = 0

    def __init__(self, name, balance):
        self.name = name
        self.account = self.gen_account()
        self.balance = balance
        self.deposit_count = 0
        self.deposit_record = []
        self.withdraw_record = []
        Account.accounts += 1

    @classmethod
    def get_account_num(cls):
        return cls.accounts

    def deposit(self, amount):
        while True:
            deposit = int(input('Enter the amount you wish to deposit. : '))
            if deposit > 0:
                self.balance += deposit
                self.deposit_count += 1
                self.deposit_record.append(str(amount))

            if self.deposit_count == 5:
                self.balance *= 1.01
            else:
                print('The deposit amount can be at least 1 won.')

            while True:
                additional = int(input('Would you like to deposit more? Please enter 1 for yes or no for 2. : '))
                if additional == 1:
                    break
                elif additional == 2:
                    break
                else:
                    print("Please enter 1 for yes or no for 2.")
                if additional == 1:
                    continue
                else:
                    break

    def withdraw(self, withdraw):
        while True:
            withdraw = int(input("Enter the amount you wish to withdraw. : "))

            if self.balance >= withdraw:
                self.balance -= withdraw
                self.withdraw_record.append(-withdraw)
            else:
                print('Withdrawals cannot exceed the account balance.')

            while True:
                additional_withdraw = int(input('Would you like to withdraw more? Please enter 1 for yes or no for 2. : '))

                if additional_withdraw == 1:
                    break
                elif additional_withdraw == 2:
                    break
                else:
                    print("Please enter 1 for yes or no for 2.")
                if additional_withdraw == 1:
                    continue
                else:
                    break

    def gen_account(self):
        account_number = "{}-{}-{}".format(''.join(map(str, [r.randint(0, 9) for _ in range(3)])), ''.join(map(str, [r.randint(0, 9) for _ in range(2)])), ''.join(map(str, [r.randint(0, 9) for _ in range(6)])))
        return account_number

    def display_info(self):
        formatted_balance = '{:,}'.format(self.balance)
        print(f'은행이름: {self.bank}, 예금주: {self.name}, 계좌번호: {self.account}, 잔고: {formatted_balance} krw')

a = Account("Lezzy", 1000000)
b = Account("Mark", 580000)
c = Account("Donald", 640000)

account_list = [a, b, c]

print("\nCustomer Info: ")
for account in account_list:
    account.display_info()

print(f'\nTotal number of accounts: {Account.get_account_num()}')

vip = [account for account in account_list if account.balance >= 1000000]

print("\nVIP Info: ")
for c in vip:
    c.display_info()
