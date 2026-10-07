/*
 * Delta Robot - TELEOPERATION + AUTONOMOUS firmware (Teensy 4.1)
 *
 * Leader   : 3x AS5600 magnetic encoders on a hand-moved leader arm.
 * Follower : 3x Feetech STS3215 servos on Serial1 (1 Mbit/s).
 * In RECORD mode the Teensy reads the leader angles, commands the servos to match, reads the
 * servos back, and streams both over USB serial as CSV. In AUTO mode the host (the policy)
 * sends joint targets + the magnet command instead.
 *
 * Serial protocol (USB, 115200, newline-terminated) -- used by robot/record_demos.py,
 * robot/run_policy.py and robot/teensy_check.py:
 *
 *   Host -> Teensy:
 *     s                     start RECORD mode  (servos mirror the leader)
 *     a                     start AUTONOMOUS mode (host sends commands)
 *     c,<a1>,<a2>,<a3>[,m]  in AUTO mode: drive the 3 servos to these angles (deg); m = magnet 0/1
 *     x                     stop (go idle)
 *     d                     one-shot DEBUG dump (encoders + button/magnet state)
 *
 *   Electromagnet: a button shorting GPIO4<->GPIO5 energizes an electromagnet on the follower,
 *   driven through an H-bridge (IN1=pin30, IN2=pin32). MOMENTARY: button pressed = magnet ON
 *   (grab), released = OFF. In AUTO mode the host owns the magnet.
 *
 *   Teensy -> Host (while streaming, STREAM_HZ = 60):
 *     <millis>,<lead1>,<lead2>,<lead3>,<foll1>,<foll2>,<foll3>,<mag>
 *       - action  = leader  (fields 1..3)   <- what you commanded by hand
 *       - state   = follower(fields 4..6)   <- where the servos actually are
 *       - mag     = field 7 (0=off, 1=on)   <- electromagnet state
 *     A failed read is streamed as -999 (the host skips those frames). In AUTO mode the leader
 *     encoders are not read (they may be unplugged) and stream as -999.
 *
 * Units: leader and follower are both in DEGREES (90 deg = the calibration reference pose).
 * Every servo command is clamped to DEG_MIN..DEG_MAX.
 */

#include <SCServo.h>
#include <Wire.h>
#include <SoftWire.h>   // Steve Marple's "SoftWire" (Arduino Library Manager)

// ============================================================
//  SERVOS (follower)
// ============================================================
#define SERVO_SERIAL Serial1          // TX=pin1, RX=pin0 on Teensy 4.1
#define SERVO_BAUD   1000000
SMS_STS sms_sts;

const int   SERVO_IDS[3]        = {2, 3, 1};
const int   NINETY_DEG_UNITS[3] = {2101, 2012, 2012};   // servo units at 90 deg, per servo  *** CALIBRATE ME ***
const float UNITS_PER_DEG       = 4096.0 / 360.0;
const int   MOVE_SPEED          = 1500;                  // responsive teleop tracking
const int   MOVE_ACCEL          = 50;

int   degToUnits(float deg, int s) { return (int)round((deg - 90.0) * UNITS_PER_DEG + NINETY_DEG_UNITS[s]); }
float unitsToDeg(int u,   int s)   { return (u - NINETY_DEG_UNITS[s]) / UNITS_PER_DEG + 90.0; }

// ============================================================
//  ENCODERS (leader) -- 3x AS5600, one per I2C bus
//  All AS5600 share address 0x36, so each needs its own bus.
//  Two live on hardware I2C; one moved to a bit-banged SoftWire
//  bus after a hardware pin broke on the original wiring.
// ============================================================
#define AS5600_ADDR         0x36
#define AS5600_RAW_ANGLE_HI 0x0C     // RAW ANGLE register (0x0C hi, 0x0D lo), 12-bit

// ---- BUS ASSIGNMENT --------------------------------------  *** EDIT ME ***
// Encoders 0 and 1 on the two still-working hardware buses.
// Teensy 4.1 hardware I2C buses are: Wire, Wire1, Wire2.
TwoWire* HW_BUS[2] = { &Wire, &Wire2 };   // <-- set to your two working buses

// Encoder 2 on SoftWire.                                     *** SET THESE PINS ***
#define SOFT_SDA  20                 // <-- CHANGE to your SoftWire SDA pin
#define SOFT_SCL  21                 // <-- CHANGE to your SoftWire SCL pin
SoftWire  soft(SOFT_SDA, SOFT_SCL);
uint8_t   softTx[8], softRx[8];

// ---- PER-ENCODER CALIBRATION ------------------------------  *** CALIBRATE ME ***
// Run the 'd' command with each joint held at REF_DEG, read the raw counts it
// prints, and paste them into RAW_AT_REF. Flip a DIR sign if that joint reads
// backwards (degrees go down when the arm goes up).
int   RAW_AT_REF[3] = { 1672, 1519, 2968 };     // raw 0..4095 when the joint is at REF_DEG
int   DIR[3]        = { -1, -1, -1 };  // +1 or -1 per joint
float REF_DEG       = 90.0;            // the angle you hold each joint at to calibrate

// ---- SAFETY: clamp every servo command to a safe joint range ----
const float DEG_MIN = 30.0;
const float DEG_MAX = 110.0;

// ============================================================
//  BUTTON + ELECTROMAGNET
//  A momentary button shorts BTN_DRIVE to BTN_SENSE. BTN_DRIVE is
//  driven LOW and BTN_SENSE uses the internal pull-up, so a press
//  reads LOW. The magnet is ON while the button is held (debounced),
//  driven through an H-bridge (IN1 HIGH + IN2 LOW = ON).
// ============================================================
#define BTN_DRIVE 4          // driven LOW; button shorts this to BTN_SENSE
#define BTN_SENSE 5          // INPUT_PULLUP; reads LOW when pressed
#define MAG_IN1   30         // H-bridge IN1 (HIGH = magnet on)
#define MAG_IN2   32         // H-bridge IN2 (held LOW)
const uint32_t BTN_DEBOUNCE_MS = 40;

// Operating mode (declared here so updateButton can see it): the button owns the
// magnet in RECORD/IDLE, but the host/policy owns it in AUTO.
enum Mode { IDLE, RECORD, AUTO };
Mode mode = IDLE;

bool     magOn           = false;   // current electromagnet state
int      btnStable       = HIGH;    // debounced level (HIGH = released)
int      btnLastRead     = HIGH;
uint32_t btnLastChangeMs = 0;

void setMagnet(bool on) {
  magOn = on;
  digitalWrite(MAG_IN1, on ? HIGH : LOW);
  digitalWrite(MAG_IN2, LOW);
}

// Poll the button (debounced). Momentary: magnet follows the button —
// pressed = ON, released = OFF (no toggle / flip-flop).
void updateButton() {
  if (mode == AUTO) return;                  // in AUTO the host (policy) owns the magnet
  int r = digitalRead(BTN_SENSE);
  if (r != btnLastRead) { btnLastRead = r; btnLastChangeMs = millis(); }
  if ((millis() - btnLastChangeMs) >= BTN_DEBOUNCE_MS && r != btnStable) {
    btnStable = r;
    setMagnet(btnStable == LOW);             // pressed (LOW) = ON, released = OFF
    Serial.print("# MAGNET "); Serial.println(magOn ? "ON" : "OFF");
  }
}

// ============================================================
//  Encoder read (templated so hardware TwoWire and SoftWire
//  share one code path -- both expose the standard Wire API).
// ============================================================
template <class WIRE>
int readRawAngle(WIRE &w) {
  w.beginTransmission(AS5600_ADDR);
  w.write(AS5600_RAW_ANGLE_HI);
  if (w.endTransmission(false) != 0) return -1;          // repeated start
  if (w.requestFrom((uint8_t)AS5600_ADDR, (uint8_t)2) != 2) return -1;
  int hi = w.read();
  int lo = w.read();
  return ((hi << 8) | lo) & 0x0FFF;                       // 12-bit
}

int rawEncoder(int i) {
  if (i < 2) return readRawAngle(*HW_BUS[i]);
  return readRawAngle(soft);
}

// raw counts -> degrees, using per-encoder calibration, with wraparound.
float encoderDeg(int i) {
  int raw = rawEncoder(i);
  if (raw < 0) return NAN;
  int d = raw - RAW_AT_REF[i];
  if (d >  2048) d -= 4096;
  else if (d < -2048) d += 4096;
  return DIR[i] * d * (360.0 / 4096.0) + REF_DEG;
}

float followerDeg(int i) {
  int u = sms_sts.ReadPos(SERVO_IDS[i]);
  return (u == -1) ? NAN : unitsToDeg(u, i);
}

float clampDeg(float d) {
  if (d < DEG_MIN) return DEG_MIN;
  if (d > DEG_MAX) return DEG_MAX;
  return d;
}

void driveServos(float d0, float d1, float d2) {
  float t[3] = { clampDeg(d0), clampDeg(d1), clampDeg(d2) };
  for (int i = 0; i < 3; i++)
    sms_sts.WritePosEx(SERVO_IDS[i], degToUnits(t[i], i), MOVE_SPEED, MOVE_ACCEL);
}

// ============================================================
//  Streaming
// ============================================================
const uint32_t STREAM_HZ = 60;                 // Python takes the newest line at 30 Hz
const uint32_t STREAM_US = 1000000UL / STREAM_HZ;
uint32_t lastStreamUs = 0;

void printVal(float v) {
  if (isnan(v)) Serial.print("-999");
  else          Serial.print(v, 2);
}

void streamFrame(float l0, float l1, float l2, float f0, float f1, float f2, int mag) {
  Serial.print(millis()); Serial.print(',');
  printVal(l0); Serial.print(',');
  printVal(l1); Serial.print(',');
  printVal(l2); Serial.print(',');
  printVal(f0); Serial.print(',');
  printVal(f1); Serial.print(',');
  printVal(f2); Serial.print(',');
  Serial.println(mag);
}

// ============================================================
//  Modes + command parsing
// ============================================================

void debugDump() {
  Serial.println("# DEBUG raw encoder counts (hold joints at REF_DEG, paste into RAW_AT_REF):");
  for (int i = 0; i < 3; i++) {
    int raw = rawEncoder(i);
    Serial.print("#  enc "); Serial.print(i);
    Serial.print("  raw="); Serial.print(raw);
    Serial.print("  deg="); Serial.print(encoderDeg(i), 2);
    Serial.print("  servo_deg="); Serial.println(followerDeg(i), 2);
  }
  Serial.print("#  button="); Serial.print(digitalRead(BTN_SENSE) == LOW ? "PRESSED" : "released");
  Serial.print("  magnet=");  Serial.println(magOn ? "ON" : "OFF");
}

void handleCommand(String line) {
  line.trim();
  if (line.length() == 0) return;

  if (line == "s") { mode = RECORD; return; }
  if (line == "a") { mode = AUTO;   return; }
  if (line == "x") { mode = IDLE;   return; }
  if (line == "d") { debugDump();   return; }

  if (line.startsWith("c")) {                    // c,a1,a2,a3[,m]  (autonomous drive)
    float v[4] = { NAN, NAN, NAN, NAN };         // v[3] = optional magnet (0/1)
    int start = line.indexOf(',');
    for (int i = 0; i < 4 && start != -1; i++) {
      int nxt = line.indexOf(',', start + 1);
      String tok = (nxt == -1) ? line.substring(start + 1)
                               : line.substring(start + 1, nxt);
      v[i] = tok.toFloat();
      start = nxt;
    }
    if (!isnan(v[0]) && !isnan(v[1]) && !isnan(v[2]))
      driveServos(v[0], v[1], v[2]);
    if (!isnan(v[3]))                            // policy-controlled magnet in AUTO
      setMagnet(v[3] > 0.5);
    return;
  }
}

// ============================================================
void setup() {
  Serial.begin(115200);                 // USB to laptop
  SERVO_SERIAL.begin(SERVO_BAUD);       // servo bus
  sms_sts.pSerial = &SERVO_SERIAL;
  sms_sts.IOTimeOut = 5;                // ms. Default is 100: a servo that fails to reply
                                        // stalls the loop 100ms PER read (3/frame = 300ms).
                                        // Fail fast so noise from the magnet can't crater the rate.

  for (int i = 0; i < 2; i++) { HW_BUS[i]->begin(); HW_BUS[i]->setClock(400000); }

  soft.setTxBuffer(softTx, sizeof(softTx));
  soft.setRxBuffer(softRx, sizeof(softRx));
  soft.setDelay_us(3);
  soft.begin();

  // Button + electromagnet
  pinMode(BTN_DRIVE, OUTPUT); digitalWrite(BTN_DRIVE, LOW);   // acts as GND for the button
  pinMode(BTN_SENSE, INPUT_PULLUP);                           // reads LOW when pressed
  pinMode(MAG_IN1, OUTPUT);
  pinMode(MAG_IN2, OUTPUT);
  setMagnet(false);                                          // start with the magnet off

  delay(200);
  Serial.println("READY");
}

void loop() {
  // ---- handle incoming commands (line-based) ----
  static String buf;
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n')      { handleCommand(buf); buf = ""; }
    else if (c != '\r') { buf += c; }
  }

  updateButton();                        // poll button; magnet = pressed state (works in every mode)

  if (mode == IDLE) return;

  // ---- paced streaming ----
  uint32_t now = micros();
  if ((uint32_t)(now - lastStreamUs) < STREAM_US) return;
  lastStreamUs = now;

  float lead[3], foll[3];
  // The leader is unused in AUTO, and reading an unplugged encoder can stall the loop
  // (floating I2C lines, no timeout), so skip it there. Streams -999 for the leader.
  for (int i = 0; i < 3; i++) lead[i] = (mode == AUTO) ? NAN : encoderDeg(i);

  if (mode == RECORD) {
    // servos mirror the leader; skip any joint whose encoder failed
    for (int i = 0; i < 3; i++)
      if (!isnan(lead[i])) sms_sts.WritePosEx(SERVO_IDS[i], degToUnits(clampDeg(lead[i]), i), MOVE_SPEED, MOVE_ACCEL);
  }
  // in AUTO mode the servos are driven by incoming 'c,...' commands instead.

  for (int i = 0; i < 3; i++) foll[i] = followerDeg(i);
  streamFrame(lead[0], lead[1], lead[2], foll[0], foll[1], foll[2], magOn ? 1 : 0);
}
