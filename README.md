# تحلیل متن فارسی با پایتون

این پروژه یک ابزار تحلیل متن فارسی است که با استفاده از کتابخانه‌های مختلف پایتون، قابلیت‌های متنوعی برای پردازش و تحلیل متن‌های فارسی فراهم می‌آورد. این برنامه شامل ویژگی‌های زیر است:

- **توکن‌سازی و نرمال‌سازی متن فارسی**
- **تحلیل احساسات**
- **خلاصه‌سازی متن**
- **تصحیح خطاهای املایی**
- **تحلیل موضوعی**
- **تشخیص زبان**
- **تحلیل شبکه‌های معنایی**
- **تجزیه و تحلیل بصری**

## ویژگی‌ها

- **توکن‌سازی و نرمال‌سازی متن**: با استفاده از `hazm`.
- **تحلیل احساسات**: با استفاده از مدل‌های پیش‌آماده فارسی از `transformers`.
- **خلاصه‌سازی متن**: با استفاده از مدل‌های پیش‌آماده فارسی.
- **تصحیح خطاهای املایی**: با استفاده از `pyspellchecker`.
- **تحلیل موضوعی**: با استفاده از `gensim`.
- **تشخیص زبان**: با استفاده از `langdetect`.
- **تحلیل شبکه‌های معنوی**: با استفاده از `networkx`.
- **تجزیه و تحلیل بصری**: با استفاده از `matplotlib` و `seaborn`.

## پیش‌نیازها

برای اجرای این برنامه، نیاز به نصب کتابخانه‌های زیر دارید:

```bash
pip install hazm pyspellchecker transformers gensim langdetect networkx matplotlib seaborn

```


نحوه استفاده

	1.	تنظیم محیط: ابتدا کتابخانه‌های مورد نیاز را نصب کنید.
	2.	اجرای برنامه: برای استفاده از برنامه، فایل main.py را اجرا کنید. می‌توانید متن خود را به جای متن نمونه در کد قرار دهید.

مثال استفاده

در فایل main.py، متنی نمونه برای تحلیل فراهم شده است:


if __name__ == "__main__":
    text = "علی و مریم به پارک رفتند. مریم از دیدن گل‌ها بسیار خوشحال شد."

    result = analyze_text(text)

    print("Word Count:", result['word_count'])
    print("Sentiment:", result['sentiment'])
    print("Summary:", result['summary'])
    print("Corrected Text:", result['corrected_text'])
    print("Topics:", result['topics'])
    print("Language:", result['language'])

    visualize_data(result)


   # توضیحات

	•	توکن‌سازی و نرمال‌سازی: متن به کلمات توکن شده و نرمالیزه می‌شود.
	•	تحلیل احساسات: احساسات متن با استفاده از مدل‌های BERT فارسی تحلیل می‌شود.
	•	خلاصه‌سازی: متن به طور خودکار خلاصه می‌شود.
	•	تصحیح خطاهای املایی: خطاهای املایی متن اصلاح می‌شود.
	•	تحلیل موضوعی: موضوعات اصلی متن شناسایی می‌شود.
	•	تشخیص زبان: زبان متن تشخیص داده می‌شود.
	•	تحلیل شبکه‌های معنوی: شبکه‌های معنوی کلمات بصری‌سازی می‌شود.
	•	تجزیه و تحلیل بصری: داده‌های تحلیل شده به صورت بصری نمایش داده می‌شود.

مشارکت

اگر می‌خواهید به این پروژه کمک کنید، لطفاً تغییرات و پیشنهادات خود را به صورت Pull Request ارسال کنید. برای گزارش مشکلات یا درخواست ویژگی‌ها، لطفاً از بخش Issues استفاده کنید.