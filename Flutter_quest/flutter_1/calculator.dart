class Calculator {
  double num1 = 10;
  double num2 = 5;

  void addition() {
    double add = num1 + num2;
    print('Addition: $num1 + $num2 = $add');
  }

  void subtraction() {
    double subtraction = num1 - num2;
    print('Subtraction: $num1 - $num2 = $subtraction');
  }

  void multiplication() {
    double multiplication = num1 * num2;
    print('Multiplication: $num1 * $num2 = $multiplication');
  }

  void division() {
    if (num2 != 0) {
      double division = num1 / num2;
      print('Division: $num1 / $num2 = $division');
    } else {
      print('Cannot divide by zero!');
    }
  }
}

void main() {
  Calculator calculator = Calculator();

  calculator.addition();
  calculator.subtraction();
  calculator.multiplication();
  calculator.division();
}
