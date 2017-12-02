var tng = require('./tng');
var perplexity_tng = require('./perplexity_tng');

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
var result = tng('direct', sentences, 3, 3, ['th']);
console.log('MODEL RAW OUTPUT: ');
console.log(JSON.stringify(result.topicModel));
console.log('');
console.log('MODEL INFORMATION OUTPUT: ');
//result.printReadableOutput();

var perplexity = perplexity_tng(result.topicModel, sentences, ['th']);
console.log('Perplexity = '+perplexity);
