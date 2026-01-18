#include <Wire.h>
#include <Adafruit_INA219.h>
#include <math.h>

static const int DAC_PIN = 25;       
static const int POL_PIN = 27;       
static const bool USE_POLARITY_PIN = false;  

static const int I2C_SDA = 21;
static const int I2C_SCL = 22;

static const float DAC_VREF = 3.30f; 
static const float DAC_MAX  = 255.0f;

static float current_limit_mA = 5.0f; 

Adafruit_INA219 ina219;


static inline void setOutputVoltage(float v_set_V) {
  float v = v_set_V;

  if (USE_POLARITY_PIN) {
    if (v_set_V >= 0.0f) {
      digitalWrite(POL_PIN, HIGH);  
      v = v_set_V;
    } else {
      digitalWrite(POL_PIN, LOW);    
      v = -v_set_V;
    }
  } else {

    if (v < 0.0f) v = 0.0f;
  }

  if (v > DAC_VREF) v = DAC_VREF;

  int dac_val = (int)lroundf((v / DAC_VREF) * DAC_MAX);
  dac_val = constrain(dac_val, 0, 255);

  dacWrite(DAC_PIN, dac_val);
}

static inline void outputOff() {
  dacWrite(DAC_PIN, 0);
}

static inline float readBusVoltage_V() {

  return ina219.getBusVoltage_V();
}

static inline float readCurrent_A() {

  float i_mA = ina219.getCurrent_mA();
  return i_mA / 1000.0f;
}

static inline bool overCurrentTrip(float i_A) {
  float i_mA = fabsf(i_A) * 1000.0f;
  return (i_mA > current_limit_mA);
}


static int splitTokens(String line, String *out, int maxTokens) {
  line.trim();
  if (line.length() == 0) return 0;

  int count = 0;
  int start = 0;
  while (count < maxTokens) {
    while (start < (int)line.length() && isspace((unsigned char)line[start])) start++;
    if (start >= (int)line.length()) break;

    int end = start;
    while (end < (int)line.length() && !isspace((unsigned char)line[end])) end++;

    out[count++] = line.substring(start, end);
    start = end;
  }
  return count;
}

static void printEnd() {
  Serial.println("END");
}

static void cmdID() {
  Serial.println("ESP32_MEMRISTOR_CONTROLLER v1");
  Serial.print("DAC_PIN=");
  Serial.print(DAC_PIN);
  Serial.print(" INA219=");
  Serial.print("OK");
  Serial.print(" ILIM_mA=");
  Serial.println(current_limit_mA, 3);
  printEnd();
}

static void cmdSETLIM(float lim_mA) {
  if (lim_mA < 0.1f) lim_mA = 0.1f;
  current_limit_mA = lim_mA;
  Serial.print("OK ILIM_mA ");
  Serial.println(current_limit_mA, 3);
  printEnd();
}


static void cmdPULSE(float v_V, int width_ms, int n, int gap_ms) {
  if (n < 1) n = 1;
  if (width_ms < 1) width_ms = 1;
  if (gap_ms < 0) gap_ms = 0;

  for (int i = 0; i < n; i++) {
    setOutputVoltage(v_V);


    int half = width_ms / 2;
    if (half > 0) delay(half);

    float vbus = readBusVoltage_V();
    float i_A  = readCurrent_A();


    if (overCurrentTrip(i_A)) {
      outputOff();
      Serial.print("TRIP ");
      Serial.print(vbus, 6);
      Serial.print(" ");
      Serial.println(i_A, 9);
      printEnd();
      return;
    }


    int rem = width_ms - half;
    if (rem > 0) delay(rem);

    outputOff();
    if (gap_ms > 0) delay(gap_ms);


    Serial.print(vbus, 6);
    Serial.print(" ");
    Serial.println(i_A, 9);
  }
  printEnd();
}


static void cmdSWEEP(float start_V, float end_V, int steps, int settle_ms, bool bidirectional) {
  if (steps < 2) steps = 2;
  if (settle_ms < 0) settle_ms = 0;

  auto doSweep = [&](float a, float b, int s, bool includeLast) -> bool {
    for (int k = 0; k < s; k++) {
      if (!includeLast && (k == s - 1)) break;

      float t = (float)k / (float)(s - 1);
      float vset = a + t * (b - a);

      setOutputVoltage(vset);
      if (settle_ms > 0) delay(settle_ms);

      float vbus = readBusVoltage_V();
      float i_A  = readCurrent_A();

      if (overCurrentTrip(i_A)) {
        outputOff();
        Serial.print("TRIP ");
        Serial.print(vbus, 6);
        Serial.print(" ");
        Serial.println(i_A, 9);
        printEnd();
        return false;
      }

      Serial.print(vbus, 6);
      Serial.print(" ");
      Serial.println(i_A, 9);
    }
    return true;
  };

  if (!doSweep(start_V, end_V, steps, true)) return;

  if (bidirectional) {
    if (!doSweep(end_V, start_V, steps, false)) return;
  }

  outputOff();
  printEnd();
}


static void cmdREAD(float v_V, int settle_ms, int samples, int interval_ms) {
  if (samples < 1) samples = 1;
  if (settle_ms < 0) settle_ms = 0;
  if (interval_ms < 0) interval_ms = 0;

  setOutputVoltage(v_V);
  if (settle_ms > 0) delay(settle_ms);

  for (int i = 0; i < samples; i++) {
    float vbus = readBusVoltage_V();
    float i_A  = readCurrent_A();

    if (overCurrentTrip(i_A)) {
      outputOff();
      Serial.print("TRIP ");
      Serial.print(vbus, 6);
      Serial.print(" ");
      Serial.println(i_A, 9);
      printEnd();
      return;
    }

    Serial.print(vbus, 6);
    Serial.print(" ");
    Serial.println(i_A, 9);

    if (interval_ms > 0) delay(interval_ms);
  }

  outputOff();
  printEnd();
}


void setup() {
  Serial.begin(115200);
  delay(200);

  if (USE_POLARITY_PIN) {
    pinMode(POL_PIN, OUTPUT);
    digitalWrite(POL_PIN, HIGH); 
  }

  Wire.begin(I2C_SDA, I2C_SCL);

  if (!ina219.begin()) {
    Serial.println("ERR INA219 not found");
  } else {

    ina219.setCalibration_32V_1A();
  }

  outputOff();
  cmdID();
}

void loop() {
  if (!Serial.available()) return;

  String line = Serial.readStringUntil('\n');
  line.trim();
  if (line.length() == 0) return;

  String tok[8];
  int nTok = splitTokens(line, tok, 8);
  if (nTok == 0) return;

  String cmd = tok[0];
  cmd.toUpperCase();

  if (cmd == "ID") {
    cmdID();
    return;
  }

  if (cmd == "SETLIM" && nTok >= 2) {
    cmdSETLIM(tok[1].toFloat());
    return;
  }

  if (cmd == "PULSE" && nTok >= 4) {
    float v = tok[1].toFloat();
    int width_ms = tok[2].toInt();
    int count = tok[3].toInt();
    int gap_ms = (nTok >= 5) ? tok[4].toInt() : 0;
    cmdPULSE(v, width_ms, count, gap_ms);
    return;
  }

  if (cmd == "SWEEP" && nTok >= 4) {
    float startV = tok[1].toFloat();
    float endV   = tok[2].toFloat();
    int steps    = tok[3].toInt();
    int settle_ms = (nTok >= 5) ? tok[4].toInt() : 30;
    bool bidi     = (nTok >= 6) ? (tok[5].toInt() != 0) : false;
    cmdSWEEP(startV, endV, steps, settle_ms, bidi);
    return;
  }

  if (cmd == "READ" && nTok >= 5) {
    float v = tok[1].toFloat();
    int settle_ms = tok[2].toInt();
    int samples   = tok[3].toInt();
    int interval_ms = tok[4].toInt();
    cmdREAD(v, settle_ms, samples, interval_ms);
    return;
  }

  Serial.print("ERR Unknown/Bad cmd: ");
  Serial.println(line);
  printEnd();
}
