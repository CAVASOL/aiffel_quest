// https://dart.dev/language/class-modifiers

abstract class IronMan {
  late String name;
  late String suitColor;

  void fly();
  void shootLasers();
  void withstandDamage();
}

class Mark50 extends IronMan {
  Mark50(String name, String suitColor) {
    this.name = name;
    this.suitColor = suitColor;
  }

  void fly() {
    print('$name is flying!');
  }

  void shootLasers() {
    print('$name is shooting repulsors!');
  }

  void withstandDamage() {
    print('$name put on $suitColor today!');
  }
}

void main() {
  var mark50 = Mark50('Mark', 'pink');

  mark50.fly();
  mark50.shootLasers();
  mark50.withstandDamage();
}
