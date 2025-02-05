const express = require('express');  
const bodyParser = require('body-parser');  
const mysql = require('mysql2/promise');  

// 1. 创建应用  
const app = express();  
app.use(bodyParser.json());  

// 2. 建立数据库连接池  
const pool = mysql.createPool({  
  host: 'yuanmeet.rwlb.rds.aliyuncs.com',     // 例如: rdsxxx.mysql.rds.aliyuncs.com  
  user: 'MarkJosn',  
  password: 'Lmz950628',  
  database: 'user'  
});  

// 3. 分步保存接口：每个步骤填写后保存至数据库  
app.post('/api/saveProfile', async (req, res) => {  
  const { userId, step, data } = req.body;  

  // 基本校验  
  if (!userId || !step || !data) {  
    return res.status(400).json({ message: '缺少必填字段 userId/step/data' });  
  }  

  try {  
    // 先检查是否已存在此 user 的记录  
    const [rows] = await pool.execute(  
      'SELECT id FROM user_profile WHERE user_id = ?',  
      [userId]  
    );  

    // 若无记录，则插入一条空记录  
    if (rows.length === 0) {  
      await pool.execute(  
        'INSERT INTO user_profile (user_id) VALUES (?)',  
        [userId]  
      );  
    }  

    // 根据 step 值选择更新的字段  
    let sql = '';  
    let params = [];  

    switch (step) {  
      case 1:  
        sql = `UPDATE user_profile  
               SET step1_name = ?, step1_wechat = ?  
               WHERE user_id = ?`;  
        params = [data.name, data.wechatNumber, userId];  
        break;  
      case 2:  
        sql = `UPDATE user_profile  
               SET step2_sex = ?  
               WHERE user_id = ?`;  
        params = [data.sex, userId];  
        break;  
      case 3:  
        sql = `UPDATE user_profile  
               SET step3_degree = ?  
               WHERE user_id = ?`;  
        params = [data.degree, userId];  
        break;  
      case 4:  
        sql = `UPDATE user_profile  
               SET step4_job = ?  
               WHERE user_id = ?`;  
        params = [data.job, userId];  
        break;  
      case 5:  
        sql = `UPDATE user_profile  
               SET step5_hobby = ?  
               WHERE user_id = ?`;  
        params = [data.hobby, userId];  
        break;  
      case 6:  
        sql = `UPDATE user_profile  
               SET step6_introduction = ?  
               WHERE user_id = ?`;  
        params = [data.introduction, userId];  
        break;  
      case 7:  
        sql = `UPDATE user_profile  
               SET step7_mbtitype = ?  
               WHERE user_id = ?`;  
        params = [data.mbtiType, userId];  
        break;  
      case 8:  
        sql = `UPDATE user_profile  
               SET step8_photos = ?  
               WHERE user_id = ?`;  
        params = [JSON.stringify(data.photos || []), userId];  
        break;  
      default:  
        return res.status(400).json({ message: 'step 超出范围(1~8)' });  
    }  

    console.log('执行的 SQL:', sql); // 打印 SQL 语句  
    console.log('参数:', params); // 打印参数  

    // 执行更新  
    await pool.execute(sql, params);  
    return res.json({ message: '保存成功' });  

  } catch (error) {  
    console.error('保存草稿失败:', error);  
    return res.status(500).json({ message: '服务器错误' });  
  }  
});  

// 4. 启动服务器  
const PORT = 3000; // 你可改成任意端口  
app.listen(PORT, () => {  
  console.log(`Server running at http://localhost:${PORT}`);  
});