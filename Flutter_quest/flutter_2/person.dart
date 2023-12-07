class Person {
  String name;
  int age;

  Person(this.name, this.age);

  void printInfo() {
    print('Hi. My name is $name, and am $age yo.');
  }
}

void main() {
  var person = Person('Luke', 90);
  person.printInfo();
}
