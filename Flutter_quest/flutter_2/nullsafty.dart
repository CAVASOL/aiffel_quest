// https://dart.dev/null-safety/understanding-null-safety

void main() {
  String? name;

  String age = '25';

  // it's dead one.
  if (name == null) {
    print('sup?');
  } else {
    print(name.length);
  }

  print(age.length);
}
