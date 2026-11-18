'use strict';
// Native browser Intl provides dates/timezones without Python, a server or network.
const assert = require('node:assert/strict');
const {getCurrentTime} = require('../ui/web/grounding.js');
let checks = 0;
const equal = (actual, expected) => {assert.deepEqual(actual, expected); checks++;};
const at = (iso, timezone) => getCurrentTime({timezone}, {now: () => new Date(iso)});

const tokyo = at('2026-11-17T15:00:00.125Z', 'Asia/Tokyo');
equal(tokyo, {source: 'computer_clock', timezone: 'Asia/Tokyo',
  local_datetime: '2026-11-18T00:00:00+09:00', utc_datetime: '2026-11-17T15:00:00.125Z',
  date: '2026-11-18', weekday: 'Wednesday', utc_offset: '+09:00'});
equal(at('2026-11-18T06:45:30Z', 'Asia/Kathmandu').local_datetime, '2026-11-18T12:30:30+05:45');
equal(at('2026-11-18T06:45:30Z', 'Pacific/Marquesas').utc_offset, '-09:30');
equal(at('2026-11-18T06:45:30Z', 'UTC').local_datetime, '2026-11-18T06:45:30+00:00');

// Spring's nonexistent hour and autumn's repeated hour are resolved from one UTC instant.
equal(at('2026-03-08T06:59:59Z', 'America/New_York').local_datetime, '2026-03-08T01:59:59-05:00');
equal(at('2026-03-08T07:00:00Z', 'America/New_York').local_datetime, '2026-03-08T03:00:00-04:00');
equal(at('2026-11-01T05:30:00Z', 'America/New_York').local_datetime, '2026-11-01T01:30:00-04:00');
equal(at('2026-11-01T06:30:00Z', 'America/New_York').local_datetime, '2026-11-01T01:30:00-05:00');
equal(at('2026-10-03T15:29:59Z', 'Australia/Lord_Howe').local_datetime, '2026-10-04T01:59:59+10:30');
equal(at('2026-10-03T15:30:00Z', 'Australia/Lord_Howe').local_datetime, '2026-10-04T02:30:00+11:00');
equal(at('2028-02-29T23:59:59Z', 'UTC').date, '2028-02-29');
equal(at('2026-12-31T23:59:59Z', 'Pacific/Kiritimati').date, '2027-01-01');

let reads = 0, zoneReads = 0;
const defaultResult = getCurrentTime({}, {now: () => {reads++; return Date.parse('2026-11-17T15:00:00Z');},
  defaultTimeZone: () => {zoneReads++; return 'Asia/Tokyo';}});
equal([reads, zoneReads, defaultResult.date], [1, 1, '2026-11-18']);
getCurrentTime({timezone: 'UTC'}, {now: () => 0, defaultTimeZone: () => {throw new Error('must not inspect default zone');}});
checks++;
for (const args of [null, [], {timezone: 123}, {timezone: ''}, {timezone: '+09:00'},
  {timezone: 'No_Such/City'}, {timezone: 'A'.repeat(101)}, {timezone: 'UTC', command: 'ignored'}]) {
  assert.throws(() => getCurrentTime(args)); checks++;
}
for (const now of [() => NaN, () => Infinity, () => new Date('invalid')]) {
  assert.throws(() => getCurrentTime({timezone: 'UTC'}, {now}), /computer clock/); checks++;
}
const live = getCurrentTime();
equal(live.source, 'computer_clock');
assert(Number.isFinite(Date.parse(live.utc_datetime)) && JSON.stringify(live).length < 400); checks++;
console.log(`PASS ${checks} clock checks`);
