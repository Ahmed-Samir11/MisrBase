import React, { useState, useEffect } from "react";
import classnames from "classnames";
import {
  Button,
  Card,
  CardHeader,
  CardBody,
  CardFooter,
  CardImg,
  CardTitle,
  Form,
  Input,
  InputGroupAddon,
  InputGroupText,
  InputGroup,
  Container,
  Row,
  Col,
  Label,
  FormGroup,
  Alert,
  Badge,
  Progress,
} from "reactstrap";
import ExamplesNavbar from "components/Navbars/ExamplesNavbar.js";
import Footer from "components/Footer/Footer.js";

export default function TeacherInput() {
  const [squares1to6, setSquares1to6] = useState("");
  const [squares7and8, setSquares7and8] = useState("");
  const [questionFocus, setQuestionFocus] = useState(false);
  const [mcAnswerFocus, setMcAnswerFocus] = useState(false);
  const [explanationFocus, setExplanationFocus] = useState(false);
  const [form, setForm] = useState({
    QuestionText: "",
    MC_Answer: "",
    StudentExplanation: "",
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const [predictions, setPredictions] = useState(null);
  const [teacher, setTeacher] = useState(null);

  React.useEffect(() => {
    document.body.classList.toggle("register-page");
    document.documentElement.addEventListener("mousemove", followCursor);
    
    // Check if teacher is signed in
    const teacherData = localStorage.getItem('teacher');
    if (teacherData) {
      setTeacher(JSON.parse(teacherData));
    }
    
    return function cleanup() {
      document.body.classList.toggle("register-page");
      document.documentElement.removeEventListener("mousemove", followCursor);
    };
  }, []);

  const followCursor = (event) => {
    let posX = event.clientX - window.innerWidth / 2;
    let posY = event.clientY - window.innerWidth / 6;
    setSquares1to6(
      "perspective(500px) rotateY(" +
        posX * 0.05 +
        "deg) rotateX(" +
        posY * -0.05 +
        "deg)"
    );
    setSquares7and8(
      "perspective(500px) rotateY(" +
        posX * 0.02 +
        "deg) rotateX(" +
        posY * -0.02 +
        "deg)"
    );
  };

  const handleChange = (e) => {
    setForm({ ...form, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError("");
    setSuccess("");
    setPredictions(null);

    try {
      const response = await fetch('http://localhost:5000/api/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(form),
      });

      const data = await response.json();

      if (data.success) {
        setSuccess("Analysis completed successfully!");
        setPredictions(data);
      } else {
        setError(data.message || "Prediction failed");
      }
    } catch (err) {
      setError("Network error. Please check your connection.");
    } finally {
      setLoading(false);
    }
  };

  const getConfidenceColor = (confidence) => {
    switch (confidence) {
      case 'high': return 'success';
      case 'medium': return 'warning';
      case 'low': return 'info';
      default: return 'secondary';
    }
  };

  const getCorrectnessColor = (correctness) => {
    return correctness === 'True' ? 'success' : 'danger';
  };

  return (
    <>
      <ExamplesNavbar />
      <div className="wrapper">
        <div className="page-header">
          <div className="page-header-image" />
          <div className="content">
            <Container>
              <Row>
                <Col className="offset-lg-0 offset-md-3" lg="8" md="10">
                  <div
                    className="square square-7"
                    id="square7"
                    style={{ transform: squares7and8 }}
                  />
                  <div
                    className="square square-8"
                    id="square8"
                    style={{ transform: squares7and8 }}
                  />
                  <Card className="card-register">
                    <CardHeader>
                      <CardImg
                        alt="..."
                        src={require("assets/img/square-purple-1.png")}
                      />
                      <CardTitle tag="h4">
                        MisrBase Analysis
                        {teacher && (
                          <small className="d-block text-muted">
                            Welcome, {teacher.name} from {teacher.school}
                          </small>
                        )}
                      </CardTitle>
                    </CardHeader>
                    <CardBody>
                      {error && (
                        <Alert color="danger" className="mb-3">
                          {error}
                        </Alert>
                      )}
                      {success && (
                        <Alert color="success" className="mb-3">
                          {success}
                        </Alert>
                      )}
                      
                      <Form className="form" onSubmit={handleSubmit}>
                        <FormGroup>
                          <Label for="QuestionText">Question Text</Label>
                          <InputGroup
                            className={classnames({
                              "input-group-focus": questionFocus,
                            })}
                          >
                            <Input
                              id="QuestionText"
                              name="QuestionText"
                              placeholder="Enter the question text"
                              type="textarea"
                              value={form.QuestionText}
                              onFocus={() => setQuestionFocus(true)}
                              onBlur={() => setQuestionFocus(false)}
                              onChange={handleChange}
                              required
                              rows="3"
                            />
                          </InputGroup>
                        </FormGroup>
                        <FormGroup>
                          <Label for="MC_Answer">Student's Selected Answer</Label>
                          <InputGroup
                            className={classnames({
                              "input-group-focus": mcAnswerFocus,
                            })}
                          >
                            <Input
                              id="MC_Answer"
                              name="MC_Answer"
                              placeholder="Enter the student's selected answer"
                              type="text"
                              value={form.MC_Answer}
                              onFocus={() => setMcAnswerFocus(true)}
                              onBlur={() => setMcAnswerFocus(false)}
                              onChange={handleChange}
                              required
                            />
                          </InputGroup>
                        </FormGroup>
                        <FormGroup>
                          <Label for="StudentExplanation">Student's Explanation</Label>
                          <InputGroup
                            className={classnames({
                              "input-group-focus": explanationFocus,
                            })}
                          >
                            <Input
                              id="StudentExplanation"
                              name="StudentExplanation"
                              placeholder="Enter the student's explanation for their answer"
                              type="textarea"
                              value={form.StudentExplanation}
                              onFocus={() => setExplanationFocus(true)}
                              onBlur={() => setExplanationFocus(false)}
                              onChange={handleChange}
                              required
                              rows="4"
                            />
                          </InputGroup>
                        </FormGroup>
                        <Button 
                          className="btn-round" 
                          color="primary" 
                          size="lg" 
                          type="submit"
                          disabled={loading}
                        >
                          {loading ? "Analyzing..." : "Analyze Student Response"}
                        </Button>
                      </Form>

                      {/* Results Section */}
                      {predictions && (
                        <div className="mt-4">
                          <h5 className="text-center mb-3">Analysis Results</h5>
                          
                          {/* Top Prediction */}
                          <Card className="mb-3">
                            <CardBody>
                              <h6>Top Prediction</h6>
                              <Badge 
                                color={getCorrectnessColor(predictions.predictions[0].correctness)}
                                className="mb-2"
                              >
                                {predictions.predictions[0].correctness === 'True' ? 'Correct' : 'Incorrect'}
                              </Badge>
                              <p className="mb-1">
                                <strong>Category:</strong> {predictions.predictions[0].category}
                              </p>
                              <p className="mb-1">
                                <strong>Misconception:</strong> {predictions.predictions[0].misconception}
                              </p>
                              <p className="mb-0">
                                <strong>Confidence:</strong> 
                                <Badge color={getConfidenceColor(predictions.predictions[0].confidence)} className="ml-2">
                                  {predictions.predictions[0].confidence}
                                </Badge>
                              </p>
                            </CardBody>
                          </Card>

                          {/* All Predictions */}
                          <Card className="mb-3">
                            <CardBody>
                              <h6>All Predictions</h6>
                              {predictions.predictions.map((pred, index) => (
                                <div key={index} className="mb-2 p-2 border rounded">
                                  <div className="d-flex justify-content-between align-items-center">
                                    <span>
                                      <Badge color={getCorrectnessColor(pred.correctness)} className="mr-2">
                                        {pred.correctness === 'True' ? 'Correct' : 'Incorrect'}
                                      </Badge>
                                      {pred.category} - {pred.misconception}
                                    </span>
                                    <Badge color={getConfidenceColor(pred.confidence)}>
                                      {pred.confidence}
                                    </Badge>
                                  </div>
                                </div>
                              ))}
                            </CardBody>
                          </Card>

                          {/* Analysis Insights */}
                          {predictions.analysis && (
                            <Card>
                              <CardBody>
                                <h6>Response Analysis</h6>
                                <Row>
                                  <Col md="6">
                                    <p><strong>Question Length:</strong> {predictions.analysis.question_length} characters</p>
                                    <p><strong>Answer Length:</strong> {predictions.analysis.answer_length} characters</p>
                                    <p><strong>Explanation Length:</strong> {predictions.analysis.explanation_length} characters</p>
                                  </Col>
                                  <Col md="6">
                                    <p><strong>Mathematical Content:</strong> 
                                      <Badge color={predictions.analysis.has_mathematical_content ? 'success' : 'secondary'} className="ml-2">
                                        {predictions.analysis.has_mathematical_content ? 'Yes' : 'No'}
                                      </Badge>
                                    </p>
                                  </Col>
                                </Row>
                              </CardBody>
                            </Card>
                          )}
                        </div>
                      )}
                    </CardBody>
                  </Card>
                </Col>
              </Row>
              <div className="register-bg" />
              <div
                className="square square-1"
                id="square1"
                style={{ transform: squares1to6 }}
              />
              <div
                className="square square-2"
                id="square2"
                style={{ transform: squares1to6 }}
              />
              <div
                className="square square-3"
                id="square3"
                style={{ transform: squares1to6 }}
              />
              <div
                className="square square-4"
                id="square4"
                style={{ transform: squares1to6 }}
              />
              <div
                className="square square-5"
                id="square5"
                style={{ transform: squares1to6 }}
              />
              <div
                className="square square-6"
                id="square6"
                style={{ transform: squares1to6 }}
              />
            </Container>
          </div>
        </div>
        <Footer />
      </div>
    </>
  );
} 