var tng = require('./tng');
var prediction_tng = require('./prediction_tng');

var sentences = [
  'คอมพิวเตอร์ เทคโนโลยี่ โลก แสดงผล',
  'โลก ต้นไม้ ธรรมชาติ ลำธาร เทคโนโลยี่ ทำลาย',
  'เทคโนโลยี่ โลก ธรรมมะ หลุดพ้น ดับทุกข์',
  'คอมพิวเตอร์ เทคโนโลยี่ โลก แสดงผล',
  'โลก ต้นไม้ ธรรมชาติ ลำธาร เทคโนโลยี่ ทำลาย',
  'เทคโนโลยี่ โลก ธรรมมะ หลุดพ้น ดับทุกข์',
  'คอมพิวเตอร์ เทคโนโลยี่ โลก แสดงผล',
  'โลก ต้นไม้ ธรรมชาติ ลำธาร เทคโนโลยี่ ทำลาย',
  'เทคโนโลยี่ โลก ธรรมมะ หลุดพ้น ดับทุกข์',
  'คอมพิวเตอร์ เทคโนโลยี่ โลก แสดงผล',
  'โลก ต้นไม้ ธรรมชาติ ลำธาร เทคโนโลยี่ ทำลาย',
  'เทคโนโลยี่ โลก ธรรมมะ หลุดพ้น ดับทุกข์',
  'คอมพิวเตอร์ เทคโนโลยี่ โลก แสดงผล',
  'โลก ต้นไม้ ธรรมชาติ ลำธาร เทคโนโลยี่ ทำลาย',
  'เทคโนโลยี่ โลก ธรรมมะ หลุดพ้น ดับทุกข์',
];

var test_data = [
  'คอมพิวเตอร์ เทคโนโลยี่ โลก',
];

var result = tng('direct', sentences, 3, 3, ['th']);

console.log(JSON.stringify(result));
//result.printReadableOutput();

var sampling = prediction_tng(result.topicModel, test_data, ['th']);
console.log('Next Word = '+result.topicModel.hypers.vocab[sampling.word]);
