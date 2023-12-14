class IronMan {
  String name;
  int powerLevel;

  IronMan(this.name, this.powerLevel);

  void shoot() {
    print('$name is shooting guns!');
  }
}

class IronManMK3 extends IronMan {
  int flyingAlt;

  IronManMK3(String name, int powerLevel, this.flyingAlt)
      : super(name, powerLevel);

  void shoot() {
    print('$name is shooting repulsors!');
  }

  void fly() {
    print('$name can fly at an altitude of $flyingAlt!');
  }
}

void main() {
  IronMan mk1 = IronMan("mk1", 10);
  IronManMK3 mk3 = IronManMK3("mk3", 20, 1000);

  mk1.shoot();
  mk3.shoot();
  mk3.fly();
}
