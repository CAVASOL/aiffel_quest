// https://dart.dev/language/mixins

mixin Flyable on Animal {
  void fly() {
    print('I am flying!');
  }
}

class Animal {
  String name;

  Animal(this.name);
}

class Bird extends Animal with Flyable {
  Bird(String name) : super(name);
}

void main() {
  var bird = Bird('Bird');
  bird.fly();
}
