import React, { useState } from "react";
import classnames from "classnames";
import { useNavigate } from "react-router-dom";
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
  Alert,
} from "reactstrap";
import ExamplesNavbar from "components/Navbars/ExamplesNavbar.js";
import Footer from "components/Footer/Footer.js";

export default function TeacherSignIn() {
  const [squares1to6, setSquares1to6] = useState("");
  const [squares7and8, setSquares7and8] = useState("");
  const [emailFocus, setEmailFocus] = useState(false);
  const [passwordFocus, setPasswordFocus] = useState(false);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const navigate = useNavigate();

  React.useEffect(() => {
    document.body.classList.toggle("register-page");
    document.documentElement.addEventListener("mousemove", followCursor);
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

  const handleSignIn = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError("");
    setSuccess("");

    try {
      const response = await fetch('http://localhost:5000/api/auth/signin', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ email, password }),
      });

      const data = await response.json();

      if (data.success) {
        setSuccess("Sign in successful! Redirecting...");
        // Store teacher info in localStorage
        localStorage.setItem('teacher', JSON.stringify(data.teacher));
        // Redirect to teacher input page after 1 second
        setTimeout(() => {
          navigate('/teacher-input');
        }, 1000);
      } else {
        setError(data.message || "Sign in failed");
      }
    } catch (err) {
      setError("Network error. Please check your connection.");
    } finally {
      setLoading(false);
    }
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
                <Col className="offset-lg-0 offset-md-3" lg="5" md="6">
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
                      <CardTitle tag="h4">Teacher Sign In</CardTitle>
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
                      <Form className="form" onSubmit={handleSignIn}>
                        <InputGroup
                          className={classnames({
                            "input-group-focus": emailFocus,
                          })}
                        >
                          <InputGroupAddon addonType="prepend">
                            <InputGroupText>
                              <i className="tim-icons icon-email-85" />
                            </InputGroupText>
                          </InputGroupAddon>
                          <Input
                            placeholder="Email"
                            type="email"
                            value={email}
                            onChange={(e) => setEmail(e.target.value)}
                            onFocus={() => setEmailFocus(true)}
                            onBlur={() => setEmailFocus(false)}
                            required
                          />
                        </InputGroup>
                        <InputGroup
                          className={classnames({
                            "input-group-focus": passwordFocus,
                          })}
                        >
                          <InputGroupAddon addonType="prepend">
                            <InputGroupText>
                              <i className="tim-icons icon-lock-circle" />
                            </InputGroupText>
                          </InputGroupAddon>
                          <Input
                            placeholder="Password"
                            type="password"
                            value={password}
                            onChange={(e) => setPassword(e.target.value)}
                            onFocus={() => setPasswordFocus(true)}
                            onBlur={() => setPasswordFocus(false)}
                            required
                          />
                        </InputGroup>
                      </Form>
                    </CardBody>
                    <CardFooter>
                      <Button 
                        className="btn-round" 
                        color="primary" 
                        size="lg"
                        onClick={handleSignIn}
                        disabled={loading}
                      >
                        {loading ? "Signing In..." : "Sign In"}
                      </Button>
                    </CardFooter>
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